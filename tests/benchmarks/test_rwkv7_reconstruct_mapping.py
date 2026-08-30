# SPDX-License-Identifier: Apache-2.0

import torch

from scripts.rwkv7.reconstruct_pth_from_hf import normalize_raw_tensor, reverse_name


def test_top_level_mapping() -> None:
    assert reverse_name("model.embeddings.weight") == ("emb.weight", False)
    assert reverse_name("lm_head.weight") == ("head.weight", False)


def test_layer_projection_mapping() -> None:
    assert reverse_name("model.layers.2.attn.r_proj.weight") == (
        "blocks.2.att.receptance.weight",
        False,
    )
    assert reverse_name("model.layers.2.ffn_norm.bias") == (
        "blocks.2.ln2.bias",
        False,
    )


def test_lora_mapping_restores_orientation() -> None:
    assert reverse_name("model.layers.2.attn.w_lora.lora.0.weight") == (
        "blocks.2.att.w1",
        True,
    )
    assert reverse_name("model.layers.2.attn.w_lora.lora.2.weight") == (
        "blocks.2.att.w2",
        True,
    )
    assert reverse_name("model.layers.2.attn.w_lora.lora.2.bias") == (
        "blocks.2.att.w0",
        False,
    )


def test_albatross_source_tensors_are_bfloat16() -> None:
    value = torch.ones(2, dtype=torch.float16)
    for key in (
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
    ):
        assert normalize_raw_tensor(key, value).dtype == torch.bfloat16
    assert normalize_raw_tensor("head.weight", value).dtype == torch.float16
