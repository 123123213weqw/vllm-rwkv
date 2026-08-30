# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

SNAPSHOT_SCHEMA_VERSION = 1
_MAGIC = b"RWKVST01"
_HEADER_LENGTH = struct.Struct(">I")
_TENSOR_NAMES = frozenset({"shift_state", "wkv_state", "elapsed"})


class SnapshotCompatibilityError(ValueError):
    """Raised when a snapshot cannot be restored into a state pool."""


@dataclass(frozen=True)
class RWKVStateDescriptor:
    model_id: str
    model_revision: str
    layer_offset: int
    num_layers: int
    hidden_size: int
    num_heads: int
    head_size: int
    shift_dtype: str
    wkv_dtype: str
    tp_size: int = 1
    tp_rank: int = 0
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    @property
    def compatibility_key(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RWKVStateDescriptor:
        try:
            descriptor = cls(**raw)
        except TypeError as exc:
            raise SnapshotCompatibilityError(
                f"Invalid RWKV state descriptor: {exc}"
            ) from exc
        if descriptor.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotCompatibilityError(
                "Unsupported RWKV state snapshot schema "
                f"{descriptor.schema_version}; expected {SNAPSHOT_SCHEMA_VERSION}."
            )
        return descriptor


@dataclass(frozen=True)
class RWKVStateSnapshot:
    descriptor: RWKVStateDescriptor
    tensors: dict[str, torch.Tensor]
    created_at: float
    payload_sha256: str

    @property
    def size_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self.tensors.values()
        )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _expected_shapes(descriptor: RWKVStateDescriptor) -> dict[str, tuple[int, ...]]:
    return {
        "shift_state": (
            descriptor.num_layers,
            2,
            descriptor.hidden_size,
        ),
        "wkv_state": (
            descriptor.num_layers,
            descriptor.num_heads,
            descriptor.head_size,
            descriptor.head_size,
        ),
        "elapsed": (1,),
    }


def _validate_tensors(
    descriptor: RWKVStateDescriptor,
    tensors: dict[str, torch.Tensor],
) -> None:
    names = set(tensors)
    if names != _TENSOR_NAMES:
        raise SnapshotCompatibilityError(
            "RWKV state tensors must be exactly "
            f"{sorted(_TENSOR_NAMES)}, got {sorted(names)}."
        )
    expected_shapes = _expected_shapes(descriptor)
    for name, expected_shape in expected_shapes.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != expected_shape:
            raise SnapshotCompatibilityError(
                f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}."
            )
        if tensor.device.type != "cpu":
            raise SnapshotCompatibilityError(f"{name} must be stored on CPU.")
        if not tensor.is_contiguous():
            raise SnapshotCompatibilityError(f"{name} must be contiguous.")
    if _dtype_name(tensors["shift_state"].dtype) != descriptor.shift_dtype:
        raise SnapshotCompatibilityError("shift_state dtype does not match descriptor.")
    if _dtype_name(tensors["wkv_state"].dtype) != descriptor.wkv_dtype:
        raise SnapshotCompatibilityError("wkv_state dtype does not match descriptor.")
    if tensors["elapsed"].dtype != torch.int32:
        raise SnapshotCompatibilityError("elapsed must use int32.")


def serialize_snapshot(
    descriptor: RWKVStateDescriptor,
    tensors: dict[str, torch.Tensor],
    *,
    created_at: float | None = None,
) -> bytes:
    cpu_tensors = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in tensors.items()
    }
    _validate_tensors(descriptor, cpu_tensors)
    payload = save_safetensors(cpu_tensors)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    header = {
        "created_at": time.time() if created_at is None else float(created_at),
        "descriptor": asdict(descriptor),
        "payload_sha256": payload_sha256,
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return _MAGIC + _HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + payload


def deserialize_snapshot(
    blob: bytes,
    *,
    expected: RWKVStateDescriptor | None = None,
) -> RWKVStateSnapshot:
    prefix_size = len(_MAGIC) + _HEADER_LENGTH.size
    if len(blob) < prefix_size or blob[: len(_MAGIC)] != _MAGIC:
        raise SnapshotCompatibilityError("Invalid RWKV state snapshot magic.")
    (header_length,) = _HEADER_LENGTH.unpack(blob[len(_MAGIC) : prefix_size])
    payload_start = prefix_size + header_length
    if payload_start > len(blob):
        raise SnapshotCompatibilityError("Truncated RWKV state snapshot header.")
    try:
        header = json.loads(blob[prefix_size:payload_start])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotCompatibilityError("Invalid RWKV state snapshot header.") from exc
    descriptor = RWKVStateDescriptor.from_dict(header.get("descriptor", {}))
    if (
        expected is not None
        and descriptor.compatibility_key != expected.compatibility_key
    ):
        raise SnapshotCompatibilityError(
            "RWKV state snapshot compatibility mismatch: "
            f"snapshot={descriptor.compatibility_key}, "
            f"target={expected.compatibility_key}."
        )
    payload = blob[payload_start:]
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != header.get("payload_sha256"):
        raise SnapshotCompatibilityError("RWKV state snapshot checksum mismatch.")
    try:
        tensors = load_safetensors(payload)
    except Exception as exc:
        raise SnapshotCompatibilityError("Invalid safetensors payload.") from exc
    _validate_tensors(descriptor, tensors)
    return RWKVStateSnapshot(
        descriptor=descriptor,
        tensors=tensors,
        created_at=float(header.get("created_at", 0.0)),
        payload_sha256=payload_sha256,
    )


__all__ = [
    "RWKVStateDescriptor",
    "RWKVStateSnapshot",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotCompatibilityError",
    "deserialize_snapshot",
    "serialize_snapshot",
]
