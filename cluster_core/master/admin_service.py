"""
Admin gRPC-сервис мастера для UI и внешних инструментов (ListWorkers).
"""
from __future__ import annotations

import grpc

from cluster_core.common.types import WorkerDescriptor, WorkerStatus
from cluster_core.master.worker_registry import WorkerRegistry
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc


_STATUS_TO_PROTO = {
    WorkerStatus.ONLINE: cluster_pb2.WORKER_STATUS_ONLINE,
    WorkerStatus.OFFLINE: cluster_pb2.WORKER_STATUS_OFFLINE,
    WorkerStatus.UNSTABLE: cluster_pb2.WORKER_STATUS_UNSTABLE,
    WorkerStatus.RECONNECTING: cluster_pb2.WORKER_STATUS_RECONNECTING,
}


def _worker_descriptor_to_proto(wd: WorkerDescriptor) -> cluster_pb2.WorkerDescriptor:
    gpus = [
        cluster_pb2.GpuInfo(
            index=g.index,
            name=g.name,
            total_vram_mb=g.total_vram_mb,
            compute_capability=g.compute_capability or "",
            backend=g.backend,
        )
        for g in wd.resources.gpus
    ]
    resources = cluster_pb2.ResourceInfo(
        cpu_cores=wd.resources.cpu_cores,
        ram_total_mb=wd.resources.ram_total_mb,
        ram_available_mb=wd.resources.ram_available_mb,
        gpus=gpus,
        torch_version=wd.resources.torch_version or "",
        cuda_version=wd.resources.cuda_version or "",
        rocm_version=wd.resources.rocm_version or "",
    )
    return cluster_pb2.WorkerDescriptor(
        id=cluster_pb2.WorkerId(host=wd.worker_id.host, port=wd.worker_id.port),
        status=_STATUS_TO_PROTO.get(wd.status, cluster_pb2.WORKER_STATUS_OFFLINE),
        resources=resources,
    )


class MasterAdminService(cluster_pb2_grpc.MasterAdminServiceServicer):
    """Сервис админ-API мастера: список воркеров для UI."""

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry

    def ListWorkers(
        self,
        request: cluster_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.WorkerList:
        workers = [
            _worker_descriptor_to_proto(wd)
            for wd in self._registry.all().values()
        ]
        return cluster_pb2.WorkerList(workers=workers)
