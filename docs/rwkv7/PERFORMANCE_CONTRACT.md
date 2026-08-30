# RWKV-7 TPS Performance Contract

## Release invariant

A release candidate passes only when its median model-only steady decode TPS
is not lower than the pinned Albatross reference in every declared case:

```text
vLLM TPS / Albatross TPS >= 1.00
```

The comparison is fail-closed. Missing measurements, a different checkpoint,
an unpinned Albatross tree, or any ratio below 1.00 blocks the release.

## Fair-comparison controls

Both paths must use all of the following identical controls:

- physical GPU and power/clock policy;
- raw RWKV-7 `.pth` checkpoint and SHA-256;
- FP16 weights and WKV mode;
- `B x T` case;
- warmup count and timed iteration count;
- paired trial count and alternating execution order;
- logits included in the timed region;
- CUDA Graph replay for steady decode.

Model loading, checkpoint preprocessing, tokenizer work, and process startup are
excluded from the steady-state kernel comparison. Server throughput is recorded
separately because Albatross is a model-loop reference, not an API scheduler.

## Default matrix

The default release matrix is `1x1,16x1,64x1`. Override `CASES` only to add
coverage. Removing a default case requires a documented hardware constraint.
Use at least 10 warmups and 30 timed iterations for publishable evidence.
The release runner defaults to four paired trials. Odd trials execute
Albatross first and even trials execute vLLM first; the gate compares the
median TPS for each side so clock drift, thermals, and launch order cannot
decide a sub-percent result.

## Reproduction

```bash
export MODEL=/models/rwkv7-g1g-1.5b-20260526-ctx8192.pth
export ALBATROSS_ROOT=/opt/Albatross
export CASES=1x1,16x1,64x1
export WARMUP=10
export ITERS=30
export TRIALS=4
bash scripts/rwkv7/run_tps_gate.sh
```

Reports are written under `evidence/rwkv7/<UTC timestamp>/`. The script exits
zero only when every case reports `overall_status=passed`.

For a source checkout paired with precompiled base vLLM extensions, build only
the RWKV product delta with:

```bash
TORCH_CUDA_ARCH_LIST=8.9 .venv/bin/python scripts/rwkv7/build_ops.py --verbose
```

## Dynamic-state runner lane

The runner lane exercises vLLM request ownership separately from the hard
Albatross model-loop gate. Each iteration creates fresh request IDs, prefills
them, performs cached recurrent decode, and releases their state rows:

```bash
.venv/bin/python benchmarks/rwkv7/benchmark_faster3a.py \
  --repo-root "$PWD" \
  --model "$MODEL" \
  --measure-vllm-runner \
  --runner-batch-size 16 \
  --runner-prompt-len 32 \
  --runner-decode-tokens 64 \
  --runner-warmup 2 \
  --runner-iters 5 \
  --measurement-output dynamic-state.json
```

Publish the runner TPS together with its state-movement counters. A valid
steady-decode result must be positive and must report zero
`resident_to_decode_copies`. Runner TPS is not divided by Albatross TPS because
Albatross has no equivalent vLLM scheduler or request lifecycle.

## Interpretation

The hard TPS claim applies to the model-only steady decode lane defined above.
End-to-end OpenAI API throughput, TTFT, cancellation, and multi-request fairness
are additional serving metrics and must not be substituted for the raw ABL
comparison.
