from __future__ import annotations

import psutil
from typing import Iterator

import grpc
import torch

from cluster_core.common.types import GpuInfo, ResourceInfo, WorkerDescriptor, WorkerId, WorkerStatus

from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc


def _detect_resources() -> ResourceInfo:
    cpu_cores = psutil.cpu_count(logical=True) or 0
    vm = psutil.virtual_memory()
    ram_total_mb = int(vm.total / (1024 * 1024))
    ram_available_mb = int(vm.available / (1024 * 1024))

    gpus: list[GpuInfo] = []
    if torch.cuda.is_available():
        num = torch.cuda.device_count()
        for idx in range(num):
            props = torch.cuda.get_device_properties(idx)
            total_vram_mb = int(props.total_memory / (1024 * 1024))
            gpus.append(
                GpuInfo(
                    index=idx,
                    name=props.name,
                    total_vram_mb=total_vram_mb,
                    compute_capability=f"{props.major}.{props.minor}",
                    backend="cuda",
                )
            )

    return ResourceInfo(
        cpu_cores=cpu_cores,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        gpus=gpus,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.version.cuda is not None else None,
        rocm_version=None,
    )


class WorkerService(cluster_pb2_grpc.WorkerServiceServicer):
    """
    Базовая реализация gRPC‑сервиса воркера.
    Пока реализует:
    - GetStatus: возвращает ресурсы узла.
    - InitShard: заглушка инициализации шарда.
    - HealthStream: echo‑heartbeat.
    - RunStage: passthrough тензора (заглушка pipeline).
    """

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._resources = _detect_resources()

    # Helpers to convert internal dataclasses -> protobuf messages.

    def _to_worker_id(self) -> cluster_pb2.WorkerId:
        return cluster_pb2.WorkerId(host=self._host, port=self._port)

    def _to_resource_info(self) -> cluster_pb2.ResourceInfo:
        gpu_msgs = [
            cluster_pb2.GpuInfo(
                index=g.index,
                name=g.name,
                total_vram_mb=g.total_vram_mb,
                compute_capability=g.compute_capability or "",
                backend=g.backend,
            )
            for g in self._resources.gpus
        ]
        return cluster_pb2.ResourceInfo(
            cpu_cores=self._resources.cpu_cores,
            ram_total_mb=self._resources.ram_total_mb,
            ram_available_mb=self._resources.ram_available_mb,
            gpus=gpu_msgs,
            torch_version=self._resources.torch_version or "",
            cuda_version=self._resources.cuda_version or "",
            rocm_version=self._resources.rocm_version or "",
        )

    # RPC methods

    def GetStatus(
        self,
        request: cluster_pb2.WorkerId,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.WorkerDescriptor:
        # Игнорируем request, так как воркер знает свой host/port.
        return cluster_pb2.WorkerDescriptor(
            id=self._to_worker_id(),
            status=cluster_pb2.WORKER_STATUS_ONLINE,
            resources=self._to_resource_info(),
        )

    def InitShard(
        self,
        request: cluster_pb2.InitShardRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.InitShardResponse:
        # Пока заглушка: просто подтверждаем инициализацию.
        # В следующих этапах здесь будет загрузка веса модели/шарда.
        return cluster_pb2.InitShardResponse(ok=True, error="")

    def HealthStream(
        self,
        request_iterator: Iterator[cluster_pb2.HealthPing],
        context: grpc.ServicerContext,
    ) -> Iterator[cluster_pb2.HealthPong]:
        for ping in request_iterator:
            yield cluster_pb2.HealthPong(
                id=self._to_worker_id(),
                nonce=ping.nonce,
                status=cluster_pb2.WORKER_STATUS_ONLINE,
            )

    def RunStage(
        self,
        request_iterator: Iterator[cluster_pb2.StageRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[cluster_pb2.StageResponse]:
        # Заглушка: просто возвращаем тот же тензор, помечая как последний пакет.
        for req in request_iterator:
            yield cluster_pb2.StageResponse(
                request_id=req.request_id,
                tensor=req.tensor,
                is_last=req.is_last,
                error="",
            )

