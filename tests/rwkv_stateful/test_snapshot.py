# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.rwkv_stateful.snapshot import (
    RWKVStateDescriptor,
    SnapshotCompatibilityError,
    deserialize_snapshot,
    serialize_snapshot,
)


def descriptor(**overrides: object) -> RWKVStateDescriptor:
    values = {
        "model_id": "rwkv7-test",
        "model_revision": "abc123",
        "layer_offset": 0,
        "num_layers": 2,
        "hidden_size": 8,
        "num_heads": 2,
        "head_size": 4,
        "shift_dtype": "float16",
        "wkv_dtype": "float32",
    }
    values.update(overrides)
    return RWKVStateDescriptor(**values)


def tensors() -> dict[str, torch.Tensor]:
    return {
        "shift_state": torch.arange(32, dtype=torch.float16).reshape(2, 2, 8),
        "wkv_state": torch.arange(64, dtype=torch.float32).reshape(2, 2, 4, 4),
        "elapsed": torch.tensor([17], dtype=torch.int32),
    }


def test_snapshot_round_trip_is_exact() -> None:
    expected = descriptor()
    snapshot = deserialize_snapshot(
        serialize_snapshot(expected, tensors(), created_at=123.5), expected=expected
    )
    assert snapshot.descriptor == expected
    assert snapshot.created_at == 123.5
    for name, tensor in tensors().items():
        torch.testing.assert_close(snapshot.tensors[name], tensor, rtol=0, atol=0)


def test_snapshot_rejects_checksum_tampering() -> None:
    blob = bytearray(serialize_snapshot(descriptor(), tensors()))
    blob[-1] ^= 1
    with pytest.raises(SnapshotCompatibilityError, match="checksum"):
        deserialize_snapshot(bytes(blob))


def test_snapshot_rejects_incompatible_model_revision() -> None:
    blob = serialize_snapshot(descriptor(), tensors())
    with pytest.raises(SnapshotCompatibilityError, match="compatibility mismatch"):
        deserialize_snapshot(blob, expected=descriptor(model_revision="different"))
