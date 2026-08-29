#!/usr/bin/env bash
# Qualitaetsabstand der Quantisierungen, gemessen gegen den FP8-Anker.
# Engine, Topologie und Tokenizer sind konstant; einzige Variable ist die
# Quantisierung. K=0, weil MTP die Verteilung nicht veraendert (und der Test
# ohnehin nicht generiert).
set -uo pipefail
REPO=/home/mp/Projekte/v100-skinny
BENCH=/home/mp/Projekte/vllm-bench
OUT=$BENCH/quality-kl.jsonl
REF=$BENCH/quality-ref-fp8.json
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
teardown() {
  rm -f "$REPO/serve-qual.pid"
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
boot() {  # $1=label $2=snapshot
  wait_gpus_free || return 1
  ( cd "$REPO" && setsid env TP=2 PORT=8020 K=0 GATE_SOFT=1 \
      DISABLE_CAR=1 NCCL_P2P_DISABLE=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,4 \
      CUDA_HOME=$REPO/.cuda-nvcc-deb/usr/local/cuda-12.8 \
      LOG=$REPO/qual-$1.log PIDFILE=$REPO/serve-qual.pid \
      bash scripts/serve-qwen38-mini.sh "$2" > "$REPO/qual-$1.boot" 2>&1 < /dev/null & )
  for _ in $(seq 1 120); do
    curl -sf -o /dev/null --max-time 3 http://127.0.0.1:8020/v1/models && return 0
    grep -qE "SERVER DIED|TIMEOUT" "$REPO/qual-$1.boot" 2>/dev/null && return 1
    sleep 10
  done
  return 1
}

echo "### FP8-Referenz" >&2
if boot fp8 "$FP8"; then
  python3 "$BENCH/quality_kl.py" --url http://127.0.0.1:8020 --model qwen3.8-27b \
      --label fp8 --dump "$REF" >> "$OUT" 2>&1
else
  echo '{"label":"fp8","status":"boot_failed"}' >> "$OUT"
fi
teardown

for pair in "nvfp4 $NVFP4" "awq $AWQ"; do
  set -- $pair
  echo "### $1" >&2
  if boot "$1" "$2"; then
    python3 "$BENCH/quality_kl.py" --url http://127.0.0.1:8020 --model qwen3.8-27b \
        --label "$1" --against "$REF" >> "$OUT" 2>&1
  else
    echo "{\"label\":\"$1\",\"status\":\"boot_failed\"}" >> "$OUT"
  fi
  teardown
done
echo "QUALITY FERTIG" >> "$OUT"
