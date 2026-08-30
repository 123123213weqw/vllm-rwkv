# Continuous Asynchronous Request Churn

Status: **PASSED**. The OpenAI-compatible vLLM server completed both randomized
arrival/departure workloads without an HTTP error or engine traceback.

| Workload | Requests | Arrival rate | Prompt | Output | Cancelled | Errors | Output TPS | TTFT p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moderate | 64 | 15/s | 8–64 | 8–32 | 10 | **0** | 230.04 | 189.71 ms | 585.72 ms |
| stress | 256 | 80/s | 8–128 | 32–128 | 42 | **0** | 1,184.48 | 10.47 s | 12.29 s |
| stateful-router smoke | 32 | 20/s | 8–32 | 8–24 | 5 | **0** | 226.67 | 587.83 ms | 686.24 ms |

The stress case generated 19,206 output tokens in 16.21 seconds. The server was
configured with `--max-num-seqs 32`; excess arrivals remained queued, which is
visible in TTFT rather than being hidden from the end-to-end measurement.

## What this validates

- Poisson arrivals continuously add requests instead of locking one batch size.
- Mixed prompt/output lengths cause rows to finish at different times.
- Deterministic early stream disconnects exercise cancellation and row release.
- The stress workload deliberately arrives faster than the engine can complete
  requests, exercising queued admission and repeated state-row reuse.
- All 320 request lifecycles completed or were deliberately cancelled; none
  returned an error and the engine log contained no error/traceback.
- A separate 32-request run used the real stateful router in front of the real
  RWKV vLLM server while its cloud endpoint was unavailable. All 32 requests
  routed locally, the Prometheus counter recorded 32 successes, in-flight
  returned to zero, and neither process logged an error.

## Metric boundary

These are end-to-end HTTP/SSE results, including tokenization, queuing, prefill,
decode, sampling, streaming, and client overhead. They are not directly
comparable to the locked model-only or runner-decode TPS gates in the parent
directory. Completed requests use the server's final OpenAI usage count;
cancelled requests count received content chunks because the client disconnects
before a final usage frame.

## Reproduction

Server:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_RAPID_SAMPLER=0 \
python -m vllm.entrypoints.cli.main serve \
  /models/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
  --host 127.0.0.1 --port 8001 --served-model-name rwkv7-test \
  --max-num-seqs 32 --gpu-memory-utilization 0.8 --enforce-eager
```

Stress workload:

```bash
python benchmarks/rwkv7/benchmark_continuous_churn.py \
  --endpoint http://127.0.0.1:8001/v1/chat/completions \
  --model rwkv7-test --requests 256 --arrival-rate 80 \
  --session-pool 64 --prompt-min 8 --prompt-max 128 \
  --output-min 32 --output-max 128 --cancel-fraction 0.15 \
  --seed 20260831 --output report-stress.json
```

Controls match the parent evidence directory: RTX 4090 with graphics clock
locked at 2520 MHz, PyTorch 2.11.0+cu128, CUDA 12.8, driver 550.142, and the
same SHA-256-pinned 1.5B checkpoint. The three JSON files contain the full
workload configuration, summary, and per-request observations.
