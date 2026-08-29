#!/usr/bin/env bash
# Formatvergleich am selben Modell (Qwen3.8-27B), selbe Topologie (2x V100 TP=2).
# Je Format K=0 (Format+Engine pur) und k=7 (Produktionswert).
# Serielle Ausfuehrung: alle Konfigurationen brauchen dieselben GPUs.
set -uo pipefail
REPO=/home/mp/Projekte/v100-skinny
BENCH=/home/mp/Projekte/vllm-bench
OUT=$BENCH/format-matrix.jsonl
: > "$OUT"

hub=/home/mp/.cache/huggingface/hub
NVFP4=$(ls -d $hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/)
AWQ=$(ls -d $hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/*/)
FP8=$(ls -d $hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/)

wait_gpus_free() {
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,4 | sort -n | tail -1)
    [ "${used:-0}" -lt 200 ] && return 0
    sleep 4
  done
  return 1
}

# Teardown ueber die GPU-Belegung statt ueber die PID-Datei: der Serve-Skript-
# Launcher beendet sich bei GATE_SOFT selbst, die Worker leben weiter, und eine
# veraltete PID-Datei laesst dann Waisen zurueck, die den naechsten Lauf blockieren.
# Der produktive llama-server (Vigilantia/llama-swap) wird explizit verschont.
teardown() {
  rm -f "$REPO/serve-matrix.pid"
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
  wait_gpus_free
}

run_one() {  # $1=label $2=snapshot $3=K
  local label="$1" snap="$2" k="$3"
  echo "### $label (K=$k) — Boot" >&2
  wait_gpus_free || { echo "{\"label\":\"$label\",\"k\":$k,\"status\":\"gpus_busy\"}" >> "$OUT"; return; }
  ( cd "$REPO" && setsid env TP=2 PORT=8020 K="$k" GATE_SOFT=1 \
      DISABLE_CAR=1 NCCL_P2P_DISABLE=1 ASYNC_SCHED=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,4 \
      CUDA_HOME=$REPO/.cuda-nvcc-deb/usr/local/cuda-12.8 \
      LOG=$REPO/matrix-$label-k$k.log PIDFILE=$REPO/serve-matrix.pid \
      bash scripts/serve-qwen38-mini.sh "$snap" > "$REPO/matrix-$label-k$k.boot" 2>&1 < /dev/null & )
  for _ in $(seq 1 120); do
    grep -qE "READY|GATE_SOFT|SERVER DIED|TIMEOUT" "$REPO/matrix-$label-k$k.boot" 2>/dev/null && break
    sleep 10
  done
  if ! curl -sf -o /dev/null --max-time 5 http://127.0.0.1:8020/v1/models; then
    echo "{\"label\":\"$label\",\"k\":$k,\"status\":\"boot_failed\"}" >> "$OUT"
    teardown; return
  fi
  # Kohaerenz-Stichprobe (bekannte Loesung 391) + Messung
  local ans
  ans=$(curl -sf --max-time 300 http://127.0.0.1:8020/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Was ist 17 mal 23? Antworte in einem Satz."}],"temperature":0,"max_completion_tokens":60,"chat_template_kwargs":{"enable_thinking":false}}' \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].replace(chr(10)," ")[:120])' 2>/dev/null)
  python3 "$BENCH/bench.py" --url http://127.0.0.1:8020 --model qwen3.8-27b \
      --label "$label-k$k" --cells math code --seeds 1001 2002 --max-tokens 1024 \
      >> "$OUT" 2>&1
  local acc
  acc=$(curl -s http://127.0.0.1:8020/metrics | grep -E "spec_decode_num_(draft_tokens|accepted_tokens)_total\{" | awk '{print $2}' | tr '\n' '/')
  echo "{\"label\":\"$label\",\"k\":$k,\"status\":\"ok\",\"probe\":\"$ans\",\"draft_accepted\":\"$acc\"}" >> "$OUT"
  teardown
}

for k in 0 7; do
  run_one nvfp4 "$NVFP4" "$k"
  run_one awq   "$AWQ"   "$k"
  run_one fp8   "$FP8"   "$k"
done
echo "MATRIX FERTIG" >> "$OUT"
