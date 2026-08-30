#!/usr/bin/env python3
"""Reconstruct an Albatross-compatible RWKV-7 state dict from HF shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


_TOP_LEVEL = {
    "model.embeddings.weight": "emb.weight",
    "model.norm.weight": "ln_out.weight",
    "model.norm.bias": "ln_out.bias",
    "lm_head.weight": "head.weight",
}
_PROJECTIONS = {
    "r_proj": "receptance",
    "k_proj": "key",
    "v_proj": "value",
    "g_norm": "ln_x",
    "o_proj": "output",
}
_MODULES = {
    "attn": "att",
    "ffn": "ffn",
    "pre_norm": "ln0",
    "attn_norm": "ln1",
    "ffn_norm": "ln2",
}

# Albatross' fused embedding + ln0 preprocessing kernel accepts BF16 input
# and emits FP16. Converted HF checkpoints may store every tensor as FP16, so
# preserving shard dtypes verbatim produces a raw checkpoint that the pinned
# faster3a_2605 baseline cannot load. These are the only tensors consumed
# before Albatross converts the remaining model weights to FP16.
_ALBATROSS_BF16_KEYS = {
    "emb.weight",
    "blocks.0.ln0.weight",
    "blocks.0.ln0.bias",
}


def reverse_name(name: str) -> tuple[str, bool]:
    """Return the raw checkpoint key and whether its tensor must transpose."""
    if name in _TOP_LEVEL:
        return _TOP_LEVEL[name], False

    match = re.fullmatch(r"model\.layers\.(\d+)\.([^.]+)\.(.+)", name)
    if match is None:
        raise KeyError(f"unexpected converted RWKV-7 key: {name}")
    layer, module, suffix = match.groups()
    if module not in _MODULES:
        raise KeyError(f"unexpected RWKV-7 module in key: {name}")
    raw_module = _MODULES[module]

    lora = re.fullmatch(r"([wvag])_lora\.lora\.(0\.weight|2\.weight|2\.bias)", suffix)
    if lora is not None:
        kind, parameter = lora.groups()
        raw_suffix = {
            "0.weight": f"{kind}1",
            "2.weight": f"{kind}2",
            "2.bias": f"{kind}0",
        }[parameter]
        return f"blocks.{layer}.{raw_module}.{raw_suffix}", parameter != "2.bias"

    if module == "attn":
        head, dot, tail = suffix.partition(".")
        if head in _PROJECTIONS:
            suffix = _PROJECTIONS[head] + (dot + tail if dot else "")
    return f"blocks.{layer}.{raw_module}.{suffix}", False


def normalize_raw_tensor(raw_name: str, tensor):
    """Restore dtype constraints required by the pinned Albatross loader."""
    if raw_name in _ALBATROSS_BF16_KEYS:
        import torch

        return tensor.to(dtype=torch.bfloat16).contiguous()
    return tensor.contiguous()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"output exists: {args.output}")
    index_path = args.model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]

    import torch
    from safetensors.torch import load_file

    state = {}
    tensor_bytes = 0
    for shard_name in sorted(set(weight_map.values())):
        shard = load_file(args.model_dir / shard_name, device="cpu")
        for converted_name, tensor in shard.items():
            raw_name, transposed = reverse_name(converted_name)
            if raw_name in state:
                raise KeyError(f"duplicate reconstructed key: {raw_name}")
            raw_tensor = tensor.t() if transposed else tensor
            raw_tensor = normalize_raw_tensor(raw_name, raw_tensor)
            state[raw_name] = raw_tensor
            tensor_bytes += raw_tensor.numel() * raw_tensor.element_size()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    report = {
        "source": str(args.model_dir),
        "output": str(args.output),
        "tensor_count": len(state),
        "tensor_bytes": tensor_bytes,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": _sha256(args.output),
    }
    manifest = args.output.with_suffix(args.output.suffix + ".reconstruction.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
