# Dynamic Recurrent-State Runner

Status: **PASSED**. The native vLLM worker completed fresh request allocation,
prefill, cached recurrent decode, and request release while keeping the
recurrent state resident on the GPU.

| Active requests | Runner TPS | p50 for 64 decode steps | Model-only retention |
| ---: | ---: | ---: | ---: |
| 1 | 330.00 | 193.94 ms | 92.39% |
| 16 | 3,542.15 | 289.09 ms | 93.57% |
| 64 | 10,728.10 | 381.80 ms | 94.20% |

All three cases reported:

```json
{
  "resident_to_decode_copies": 0,
  "decode_compactions": 0,
  "decode_compaction_rows": 0
}
```

## What is dynamic in this measurement

Every iteration creates an entirely new set of request IDs, prefills 32 tokens
per request, performs 64 recurrent decode steps, and then finishes and releases
the whole set. Two iterations warm the runner and five fresh iterations are
timed. The measured portion therefore covers 5, 80, and 320 distinct request
lifecycles for the three concurrency levels.

The zero-copy result means steady decode reads and writes the request-owned rows
directly in the resident recurrent-state pool. No resident-to-scratch copy or
decode-row compaction occurred for this full-batch lifecycle workload.

## Metric boundary

TPS is measured with CUDA events around `worker.execute_model()` for the 64
decode steps. Prefill, token sampling, collective-RPC serialization, request
creation, and request release are deliberately outside the timed region. This
is a dynamic-state **runner decode** metric, not end-to-end OpenAI API
throughput and not an asynchronous-arrival load test.

Model-only retention uses the paired model-loop medians in the parent evidence
directory. It is diagnostic rather than a release threshold because Albatross
does not provide an equivalent vLLM scheduler and request-state lifecycle.

## Controls

- GPU: NVIDIA GeForce RTX 4090, graphics clock locked at 2520 MHz
- Software: PyTorch 2.11.0+cu128, CUDA 12.8, driver 550.142
- Source revision: `0d29b95fd95e0091ad17631b8c334cd0acd35167`
- vLLM base extensions: commit `a65f93fb2e295e501b929df3c291ec89c27d39e8`
- Checkpoint: `rwkv7-g1g-1.5b-20260526-ctx8192.pth`
- Prompt/decode: 32 prompt tokens, 64 recurrent decode steps
- Sampling policy: FlashInfer and rapid sampler disabled; sampling is untimed
- Estimator: p50 of five fresh-request iterations after two warmups

See the three `b*-p32-d64.json` files, raw logs, `environment.json`, GPU clock
captures, and source-file checksums in this directory.
