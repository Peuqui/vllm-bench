#!/usr/bin/env bash
# NVFP4 gegen AWQ-INT4 auf den beiden RTX 8000 (sm75) — dieselbe Engine, dieselben
# Einstellungen wie die V100-Matrix, damit beide Tabellen vergleichbar sind.
# Der Drafter lebt auf der letzten Stufe, hier also auf einer RTX: SPEC_ATTN=TRITON_ATTN
# (FLASH_ATTN_V100 stirbt dort mit "Kernel supports only Volta GPUs").
set -uo pipefail
REPO=/home/mp/Projekte/v100-skinny
BENCH=/home/mp/Projekte/vllm-bench
OUT=$BENCH/rtx-matrix.jsonl
: > "$OUT"
hub=/home/mp/.cache/huggingface/hub
NVFP4=$(ls -d $hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/)
AWQ=$(ls -d $hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/*/)

wait_free() {
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,2 | sort -n | tail -1)
    [ "${used:-0}" -lt 200 ] && return 0
    sleep 4
  done
  return 1
}
teardown() {
  local protect; protect=$(pgrep -d, llama-server)
  for sig in TERM KILL; do
    local left=0
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      case ",$protect," in *",$pid,"*) continue;; esac
      kill -"$sig" "$pid" 2>/dev/null; left=1
    done
    [ "$left" = 0 ] && break
    sleep 8
  done
  wait_free
}
run_one() {  # $1=label $2=snapshot $3=K
  wait_free || { echo "{\"label\":\"$1\",\"k\":$3,\"status\":\"gpus_busy\"}" >> "$OUT"; return; }
  ( cd "$REPO" && setsid env TP=2 PORT=8020 K="$3" ATTN_BACKEND=AUTO SPEC_ATTN=TRITON_ATTN GATE_SOFT=1 \
      DISABLE_CAR=1 NCCL_P2P_DISABLE=1 ASYNC_SCHED=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 \
      CUDA_HOME=$REPO/.cuda-nvcc-deb/usr/local/cuda-12.8 \
      LOG=$REPO/rtx-$1-k$3.log PIDFILE=$REPO/serve-rtx.pid \
      bash scripts/serve-qwen38-mini.sh "$2" > "$REPO/rtx-$1-k$3.boot" 2>&1 < /dev/null & )
  for _ in $(seq 1 120); do
    curl -sf -o /dev/null --max-time 3 http://127.0.0.1:8020/v1/models && break
    grep -qE "SERVER DIED|TIMEOUT" "$REPO/rtx-$1-k$3.boot" 2>/dev/null && break
    sleep 10
  done
  if ! curl -sf -o /dev/null --max-time 5 http://127.0.0.1:8020/v1/models; then
    echo "{\"label\":\"$1\",\"k\":$3,\"status\":\"boot_failed\"}" >> "$OUT"; teardown; return
  fi
  local ans
  ans=$(curl -sf --max-time 300 http://127.0.0.1:8020/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Was ist 17 mal 23? Antworte in einem Satz."}],"temperature":0,"max_completion_tokens":60,"chat_template_kwargs":{"enable_thinking":false}}' \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].replace(chr(10)," ")[:100])' 2>/dev/null)
  python3 "$BENCH/bench.py" --url http://127.0.0.1:8020 --model qwen3.8-27b \
      --label "rtx-$1-k$3" --cells math code --seeds 1001 2002 --max-tokens 1024 >> "$OUT" 2>&1
  local acc; acc=$(curl -s http://127.0.0.1:8020/metrics | grep -E "spec_decode_num_(draft_tokens|accepted_tokens)_total\{" | awk '{print $2}' | tr '\n' '/')
  echo "{\"label\":\"$1\",\"k\":$3,\"status\":\"ok\",\"probe\":\"$ans\",\"draft_accepted\":\"$acc\"}" >> "$OUT"
  teardown
}
for k in 0 7; do
  run_one nvfp4 "$NVFP4" "$k"
  run_one awq   "$AWQ"   "$k"
done
echo "RTX-MATRIX FERTIG" >> "$OUT"
