<!-- markdownlint-disable MD001 MD041 -->

# RWKV-vLLM Native

This repository is an independent, performance-gated distribution of vLLM
with native RWKV-7 recurrent-state scheduling and Albatross-derived CUDA
kernels. It exists because the complete upstream implementation in
[vLLM PR #46269](https://github.com/vllm-project/vllm/pull/46269) was closed
for maintenance-demand reasons rather than technical correctness.

**Release invariant:** for the same GPU, raw `.pth` checkpoint, precision,
batch size, sequence length, warmup, and timed iterations, median steady-state
decode TPS must be **greater than or equal to 1.00x** the pinned
[BlinkDL/Albatross](https://github.com/BlinkDL/Albatross)
`faster3a_2605` baseline. A result below 1.00x exits non-zero and blocks a
release. See [the performance contract](docs/rwkv7/PERFORMANCE_CONTRACT.md).

The project does not replace Albatross kernels with a slower reimplementation.
It preserves their optimized execution path and adds vLLM V1 scheduling,
OpenAI-compatible serving, dynamic recurrent-state ownership, rapid sampling,
and cancellation-safe request isolation.

| Component | Pinned source |
| --- | --- |
| vLLM integration | `vllm-project/vllm#46269` head `58907e0` |
| Albatross reference | commit `5e941fb`, `faster3a_2605` |
| Default gate cases | `B1T1`, `B16T1`, `B64T1` |
| Required TPS ratio | `vLLM / Albatross >= 1.00` for every case |

Verified on an RTX 4090 with a locked 2520 MHz graphics clock, four balanced
paired trials, 20 warmups, and 100 timed CUDA Graph replays:

| Case | Albatross TPS | vLLM TPS | Ratio |
| --- | ---: | ---: | ---: |
| `B1T1` | 355.12 | 357.19 | **1.0058x** |
| `B16T1` | 3,758.31 | 3,785.58 | **1.0073x** |
| `B64T1` | 11,384.43 | 11,388.71 | **1.0004x** |

The complete machine-readable evidence, environment manifest, reports, and all
raw trials are committed under
[`evidence/rwkv7/rtx4090-rwkv7-1p5b-locked2520-torch211-cu128`](evidence/rwkv7/rtx4090-rwkv7-1p5b-locked2520-torch211-cu128/RESULTS.md).

Project-specific entry points:

- `scripts/rwkv7/run_tps_gate.sh` — same-machine reproducible TPS gate.
- `benchmarks/rwkv7/benchmark_faster3a.py` — measurement and JSON report.
- `scripts/rwkv7/aggregate_model_only_trials.py` — balanced median estimator.
- `docs/rwkv7/SOURCE_LINEAGE.md` — exact reuse and attribution map.
- `docs/rwkv7/GOVERNANCE.md` — small-project maintenance policy.

The remainder of this README is the original vLLM project documentation.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
