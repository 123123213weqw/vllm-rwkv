#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:?set MODEL to a local raw RWKV-7 .pth checkpoint}"
ALBATROSS_ROOT="${ALBATROSS_ROOT:?set ALBATROSS_ROOT to the pinned Albatross checkout}"
ALBATROSS_IMPL="${ALBATROSS_IMPL:-faster3a_2605}"
CASES="${CASES:-1x1,16x1,64x1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-30}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT/evidence/rwkv7/$STAMP}"

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing checkpoint: $MODEL" >&2; exit 2; }
[[ -d "$ALBATROSS_ROOT/$ALBATROSS_IMPL" ]] || {
  echo "missing Albatross implementation: $ALBATROSS_ROOT/$ALBATROSS_IMPL" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
IFS=',' read -r -a case_list <<< "$CASES"
reports=()

for case_name in "${case_list[@]}"; do
  measurement="$OUT_DIR/measurement-${case_name}.json"
  report="$OUT_DIR/report-${case_name}.json"

  "$PYTHON" "$ROOT/benchmarks/rwkv7/benchmark_faster3a.py" \
    --repo-root "$ROOT" \
    --model "$MODEL" \
    --albatross-root "$ALBATROSS_ROOT" \
    --albatross-impl "$ALBATROSS_IMPL" \
    --albatross-checkpoint "$MODEL" \
    --measure-albatross-model-only \
    --albatross-case "$case_name" \
    --albatross-warmup "$WARMUP" \
    --albatross-iters "$ITERS" \
    --measurement-output "$measurement"

  "$PYTHON" "$ROOT/benchmarks/rwkv7/benchmark_faster3a.py" \
    --repo-root "$ROOT" \
    --model "$MODEL" \
    --albatross-root "$ALBATROSS_ROOT" \
    --albatross-impl "$ALBATROSS_IMPL" \
    --albatross-checkpoint "$MODEL" \
    --measurement-json "$measurement" \
    --measure-vllm-model-only \
    --vllm-case "$case_name" \
    --vllm-warmup "$WARMUP" \
    --vllm-iters "$ITERS" \
    --measurement-output "$measurement"

  "$PYTHON" "$ROOT/scripts/rwkv7/write_model_only_report.py" \
    --benchmark "$ROOT/benchmarks/rwkv7/benchmark_faster3a.py" \
    --measurement "$measurement" \
    --output "$report"
  reports+=("$report")
done

"$PYTHON" "$ROOT/scripts/rwkv7/summarize_tps_gate.py" \
  --output "$OUT_DIR/summary.json" "${reports[@]}"
echo "RWKV-7 TPS gate passed: $OUT_DIR/summary.json"
