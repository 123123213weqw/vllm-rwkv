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
- logits included in the timed region;
- CUDA Graph replay for steady decode.

Model loading, checkpoint preprocessing, tokenizer work, and process startup are
excluded from the steady-state kernel comparison. Server throughput is recorded
separately because Albatross is a model-loop reference, not an API scheduler.

## Default matrix

The default release matrix is `1x1,16x1,64x1`. Override `CASES` only to add
coverage. Removing a default case requires a documented hardware constraint.
Use at least 10 warmups and 30 timed iterations for publishable evidence.

## Reproduction

```bash
export MODEL=/models/rwkv7-g1g-1.5b-20260526-ctx8192.pth
export ALBATROSS_ROOT=/opt/Albatross
export CASES=1x1,16x1,64x1
export WARMUP=10
export ITERS=30
bash scripts/rwkv7/run_tps_gate.sh
```

Reports are written under `evidence/rwkv7/<UTC timestamp>/`. The script exits
zero only when every case reports `overall_status=passed`.

For a source checkout paired with precompiled base vLLM extensions, build only
the RWKV product delta with:

```bash
TORCH_CUDA_ARCH_LIST=8.9 .venv/bin/python scripts/rwkv7/build_ops.py --verbose
```

## Interpretation

The hard TPS claim applies to the model-only steady decode lane defined above.
End-to-end OpenAI API throughput, TTFT, cancellation, and multi-request fairness
are additional serving metrics and must not be substituted for the raw ABL
comparison.
