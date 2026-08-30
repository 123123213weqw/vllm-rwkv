# Verified RTX 4090 TPS Gate

Status: **PASSED** (`vLLM / Albatross >= 1.00` in every declared case).

| BxT | Albatross TPS | vLLM TPS | Ratio |
| --- | ---: | ---: | ---: |
| `1x1` | 355.12 | 357.19 | **1.0058x** |
| `16x1` | 3,758.31 | 3,785.58 | **1.0073x** |
| `64x1` | 11,384.43 | 11,388.71 | **1.0004x** |

## Controls

- GPU: NVIDIA GeForce RTX 4090, graphics clock locked at 2520 MHz
- Software: PyTorch 2.11.0+cu128, CUDA 12.8, driver 550.142
- Checkpoint: `rwkv7-g1g-1.5b-20260526-ctx8192.pth`
- Checkpoint SHA-256: `a593952f515302c9b5326755cb06a40499a4e2a4e54d25eb88babca4430382b5`
- Reference: Albatross `faster3a_2605` at `5e941fb1eeb7f735a562fb5bbb30fad19adc825b`
- Measurement: logits included, CUDA Graph replay, 20 warmups, 100 timed iterations
- Estimator: median of four paired trials with balanced alternating execution order

See [`summary.json`](summary.json), the per-case reports, raw measurements, all
24 trial files, and [`environment.json`](environment.json). The result is a
model-only steady-decode contract; API scheduling throughput is a separate
serving metric.

## Dynamic recurrent-state runner

The same machine also ran fresh request allocation, 32-token prefill, 64-step
decode, and request release through the native vLLM worker:

| Active requests | Runner TPS | Model-only retention |
| ---: | ---: | ---: |
| 1 | 330.00 | 92.39% |
| 16 | 3,542.15 | 93.57% |
| 64 | 10,728.10 | 94.20% |

All cases completed with zero resident-to-decode state copies. See the
[`dynamic-state`](dynamic-state/RESULTS.md) evidence and its precise metric
boundary; it is a runner decode measurement rather than end-to-end API TPS.
