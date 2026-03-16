"""
Сериализация/десериализация torch.Tensor в proto TensorPayload для pipeline RunStage.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cluster_core.grpc import cluster_pb2


def tensor_to_payload(tensor: torch.Tensor) -> "cluster_pb2.TensorPayload":
    """Записывает тензор в proto TensorPayload (meta + data)."""
    from cluster_core.grpc import cluster_pb2

    buf = io.BytesIO()
    torch.save(tensor.cpu(), buf, _use_new_zipfile_serialization=False)
    meta = cluster_pb2.TensorMeta(
        dtype=str(tensor.dtype),
        shape=list(tensor.shape),
        device=str(tensor.device),
    )
    return cluster_pb2.TensorPayload(meta=meta, data=buf.getvalue())


def payload_to_tensor(payload: "cluster_pb2.TensorPayload", device: str | torch.device = "cpu") -> torch.Tensor:
    """Восстанавливает тензор из proto TensorPayload."""
    if not payload.data:
        raise ValueError("TensorPayload.data пустой")
    t = torch.load(io.BytesIO(payload.data), map_location=device, weights_only=False)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Ожидался torch.Tensor, получено {type(t)}")
    return t
