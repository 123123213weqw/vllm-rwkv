# vllm-rwkv

This is the installable delivery shim for the RWKV7 native vLLM distribution.
It registers `RWKV7ForCausalLM` through vLLM's supported general-plugin entry
point and exposes the edge/cloud router CLI.

```bash
uv pip install ./plugins/vllm-rwkv
vllm-rwkv-doctor
rwkv-state-router --port 8080
```

## Compatibility boundary

The model implementation is an out-of-tree-style lazy registration, but the
current recurrent-state scheduler and GPU model-state hooks are deliberate
changes in `vllm-rwkv`. Installing this wheel into an arbitrary stock
vLLM wheel is therefore rejected by the doctor instead of silently running a
semantically incorrect scheduler. The long-term upstream boundary is tracked
in `docs/rwkv7/STATEFUL_EDGE_CLOUD.md`.
