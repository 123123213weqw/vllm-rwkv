# Source Lineage

This project intentionally reuses existing work rather than creating a second
RWKV inference stack.

## vLLM integration

- Source: [vLLM PR #46269](https://github.com/vllm-project/vllm/pull/46269)
- Authoring repository: `rwkv-rs/vllm-rwkv`
- Imported head: `58907e0e1776d102fc7ea1afecbb7439ef56a11e`
- License: Apache-2.0

That work supplies the RWKV-7 model/config/tokenizer, recurrent-state-aware V1
scheduler path, raw `.pth` loader, OpenAI defaults, tool parser, rapid sampler,
tests, and benchmark harness.

## CUDA execution path

- Source: [BlinkDL/Albatross](https://github.com/BlinkDL/Albatross)
- Pinned reference commit: `5e941fb1eeb7f735a562fb5bbb30fad19adc825b`
- Reference directory: `faster3a_2605`
- License: Apache-2.0

The file-level mapping is emitted in every benchmark report by
`benchmarks/rwkv7/benchmark_faster3a.py`. CUDA sources retain SPDX headers and
repository history retains original authorship.

## Local changes

This distribution adds a strict `>=1.00x` TPS floor, a multi-case gate wrapper,
evidence format, project documentation, and maintenance policy. Performance
claims are accepted only from reproducible same-machine JSON evidence.
