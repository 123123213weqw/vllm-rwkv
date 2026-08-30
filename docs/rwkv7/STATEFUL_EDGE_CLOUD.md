# RWKV7 Stateful Edge-Cloud Inference

## Product claim

`rwkv-vllm-native` is a local-first personal-assistant inference runtime that
keeps RWKV's recurrent state close to the user and spills requests to an
OpenAI-compatible cloud endpoint when the local GPU is unhealthy, full, or the
prompt exceeds a configurable local limit.

The differentiation is not another generic GPU scheduler. It is the boundary
between a recurrent model's small, explicit state and cloud-native serving:

- vLLM supplies serving, continuous batching, OpenAI APIs, and the surrounding
  model ecosystem.
- Albatross supplies the performance baseline and optimized RWKV7 kernel path.
- RWKV native state rows replace KV-cache growth for the recurrent path.
- This project adds compatible state serialization, session placement, a
  local-first/cloud-fallback router, churn tests, and deployment/FinOps assets.

One product repository owns the integration. Upstream projects remain pinned
dependencies or lineage references; they are not copied into a collection of
long-lived forks.

## Architecture

```text
personal assistant / OpenAI client
              |
              | X-RWKV-Session-ID
              v
      state-aware router :8080
       /          |           \
 SQLite       Prometheus       session placement
 registry      /metrics        + state reference
       \          |           /
        +---- routing policy --+
              /        \
             v          v
 local vLLM/RWKV7     cloud OpenAI-compatible RWKV7
 workstation GPU      KServe/Kubernetes GPU pool
```

The policy is sticky for an existing healthy session, local-first for a new
session, and cloud-fallback when local utilization crosses the configured
threshold. It emits route, fallback, latency, in-flight, and state-transfer
metrics.

## Implemented interfaces

### Recurrent-state snapshot

`RWKV7ModelState.snapshot_request(req_id)` and
`restore_request(req_id, blob)` serialize exactly three tensors for one
allocated request row:

| Tensor | Shape |
| --- | --- |
| shift state | `[local_layers, 2, hidden_size]` |
| WKV state | `[local_layers, local_heads, head_size, head_size]` |
| elapsed position | `[1]` |

The envelope uses safetensors rather than pickle, includes a SHA-256 checksum,
and rejects a restore if model ID/revision, layer partition, tensor parallel
rank/size, dimensions, dtypes, or schema version differ. Snapshot/restore must
run while that request row is quiescent.

The raw recurrent-state size is:

```text
layers * (2 * hidden_size * sizeof(shift_dtype)
          + heads * head_size^2 * sizeof(wkv_dtype)) + 4 bytes
```

This is bounded by model shape rather than conversation length, which is the
property the edge/cloud design exploits.

### Registry and object store

- `SessionRegistry` is SQLite/WAL with compare-and-swap generations and
  expiring ownership leases.
- `FilesystemSnapshotStore` is content-addressed, uses atomic writes and
  `0600` object/manifest permissions, verifies checksums, and prevents URI path
  traversal.
- The file store is suitable for one machine or a shared volume. Cross-region
  deployment should implement the same object contract on S3/OSS/COS and issue
  short-lived signed references.

### Router protocol

The router proxies:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/responses`

Request headers:

| Header | Meaning |
| --- | --- |
| `X-RWKV-Session-ID` | Stable assistant/conversation placement key |
| `X-RWKV-Force-Cloud: true` | Explicit cloud override |

Response headers:

| Header | Meaning |
| --- | --- |
| `X-RWKV-Session-ID` | Effective session key |
| `X-RWKV-Route` | `local` or `cloud` |
| `X-RWKV-Route-Reason` | Policy decision explanation |

When a stored snapshot is moved and `RWKV_STATE_TRANSFER_SUPPORTED=1`, the
router forwards
`X-RWKV-State-URI`, `X-RWKV-State-SHA256`, and
`X-RWKV-State-Generation` to the target. This flag defaults off until the
worker-control adapter described below is deployed at both endpoints.

### Current integration boundary

Tensor serialization/restore, storage, registry, routing, fallback, metrics,
and manifests are implemented and unit-testable. A completed OpenAI request is
still removed by the vLLM engine. Therefore automatic **cross-request live
resume** requires the next adapter step: a worker-control RPC that quiesces a
request before removal, persists its row, allocates the destination row, then
calls `restore_request`. The header protocol deliberately isolates that step.
Until it lands, the router provides sticky placement/fallback and the snapshot
API is a worker-level primitive, not a claim of transparent production state
migration.

## Run

### Installable plugin delivery

```bash
uv pip install ./plugins/vllm-rwkv7
rwkv-vllm-doctor
```

The plugin uses vLLM's official `vllm.general_plugins` registration point. It
checks for this distribution's recurrent scheduler/model-state hooks and
refuses arbitrary stock-vLLM installations rather than silently changing
semantics.

### Local engine plus cloud

Start a local engine:

```bash
uv run vllm serve /models/rwkv7.pth \
  --host 127.0.0.1 --port 8000 --max-num-seqs 32 --enforce-eager
```

Start the router:

```bash
export RWKV_LOCAL_URL=http://127.0.0.1:8000
export RWKV_CLOUD_URL=https://rwkv-cloud.example.com
export RWKV_CLOUD_API_KEY=...
export RWKV_SESSION_REGISTRY="$HOME/.local/share/rwkv-router/sessions.db"
uv run python -m vllm.rwkv_stateful.router --port 8080
```

Clients point their OpenAI base URL to `http://127.0.0.1:8080/v1` and attach a
stable session ID. `deploy/rwkv-stateful/docker-compose.yaml` packages the same
topology.

### Kubernetes and Serverless

```bash
helm upgrade --install rwkv deploy/rwkv-stateful/helm \
  --set model.existingClaim=rwkv-models \
  --set cloudEngine.externalUrl=https://rwkv-cloud.example.com
```

The Helm chart includes native local/cloud engine pools, the router, probes,
GPU resource requests, persistent registry storage, and an optional
ServiceMonitor. `kserve-inferenceservice.yaml` is the scale-to-zero cloud-pool
example. The SQLite registry defaults to one router replica; horizontal router
HA requires replacing it with a networked registry before increasing replicas.

## Validation

### Locked throughput gate

The existing release gate compares identical RWKV7 model-only cases against a
pinned Albatross implementation and requires `vLLM / Albatross >= 1.00` for
every B1/B16/B64 case. See `PERFORMANCE_CONTRACT.md` and committed evidence.

### Dynamic-state runner

Committed evidence exercises fresh request IDs, state allocation/release,
prefill, and decode. It records state copies/compactions in addition to TPS.

### Continuous churn

The API-level workload adds randomized arrivals and departures rather than
holding an ideal batch size constant:

```bash
uv run python benchmarks/rwkv7/benchmark_continuous_churn.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model /models/rwkv7.pth \
  --requests 1000 --arrival-rate 30 --session-pool 128 \
  --prompt-min 8 --prompt-max 512 \
  --output-min 8 --output-max 128 --cancel-fraction 0.15 \
  --output evidence/rwkv7/continuous-churn.json
```

The JSON contains per-request status, route, TTFT, latency, output count, and a
summary with p50/p95/p99 and output TPS. Early streaming disconnects exercise
cancellation-safe row release.

## FinOps loop

The decision threshold is externally configurable; it should be tuned using:

1. local GPU allocation cost and utilization from OpenCost/DCGM,
2. successful/output token counters from vLLM,
3. router cloud-fallback ratio and latency,
4. the cloud provider's token or GPU-hour bill.

The primary KPI is cost per successful output token under an SLO, not maximum
GPU utilization in isolation. Prometheus rules and example queries live under
`deploy/rwkv-stateful/monitoring`.

## Roadmap

1. Worker RPC for quiesce/snapshot/allocate/restore and true request resume.
2. S3/OSS/COS snapshot-store adapter with envelope encryption and signed URIs.
3. Redis/etcd registry backend for multi-router and multi-cluster HA.
4. Churn evidence on local-only, forced-cloud, and migration-failure scenarios.
5. Upstream scheduler-hook proposal so the model wheel can target stock vLLM.
