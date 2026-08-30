# Monitoring and FinOps

The router exposes Prometheus metrics at `/metrics`. Apply
`prometheus-rules.yaml` or enable the Helm `ServiceMonitor`.

Recommended dashboard panels:

```promql
sum by (target) (rate(rwkv_router_requests_total[5m]))
sum by (from_target, to_target) (rate(rwkv_router_fallbacks_total[5m]))
histogram_quantile(0.95, sum by (le, target) (rate(rwkv_router_request_seconds_bucket[5m])))
sum by (target) (rwkv_router_inflight_requests)
```

For OpenCost, allocate the engine and router with their standard namespace,
deployment, pod, and `app.kubernetes.io/component` labels. A useful cost KPI is:

```text
GPU allocation cost / successful output tokens
```

Join OpenCost's hourly workload allocation for the `local-engine` and
`cloud-engine` deployments with `vllm:generation_tokens_total` and compare it
to `rwkv:cloud_request_ratio:5m`. This makes the local-to-cloud threshold a
measurable FinOps control rather than a fixed architectural claim.
