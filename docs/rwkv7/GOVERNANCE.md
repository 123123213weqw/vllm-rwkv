# RWKV-vLLM Native Governance

## Scope

The project maintains the RWKV-7 serving path and a thin, pinned vLLM
distribution. It does not promise to mirror every upstream vLLM change.

## Change policy

1. Correctness changes require targeted unit tests.
2. CUDA/model hot-path changes require same-machine Albatross evidence.
3. A release cannot lower any default-matrix TPS ratio below 1.00.
4. Upstream updates are adopted only for security, compatibility, or measured
   serving value; routine rebases are not a goal.
5. Source lineage and third-party license notices must remain intact.

## Decision making

Maintainers use lazy consensus for documentation and tests. Changes to kernels,
state ownership, benchmark definitions, or release thresholds require explicit
maintainer review. The benchmark threshold may be raised but not lowered for a
release branch.

## Releases

A release contains the source commit, model SHA-256, Albatross commit, hardware
inventory, raw measurements, generated reports, and exact reproduction command.
