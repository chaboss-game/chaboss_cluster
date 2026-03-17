"""
Admin gRPC-сервис мастера для UI и внешних инструментов (ListWorkers, LoadModel).
"""
from __future__ import annotations

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
        os_name=wd.resources.os_name or "",
        os_version=wd.resources.os_version or "",
    )
    return cluster_pb2.WorkerDescriptor(
        id=cluster_pb2.WorkerId(host=wd.worker_id.host, port=wd.worker_id.port),
        status=_STATUS_TO_PROTO.get(wd.status, cluster_pb2.WORKER_STATUS_OFFLINE),
        resources=resources,
        token_fingerprint=wd.token_fingerprint or "",
        token_status=wd.token_status or "UNKNOWN",
    )


class MasterAdminService(cluster_pb2_grpc.MasterAdminServiceServicer):
    """Сервис админ-API мастера: список воркеров, загрузка модели."""

    def __init__(self, registry: WorkerRegistry, master_node: object = None) -> None:
        self._registry = registry
        self._master_node = master_node

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

    def LoadModel(
        self,
        request: cluster_pb2.LoadModelRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.LoadModelResponse:
        if not request.hf_model_id or not request.hf_model_id.strip():
            return cluster_pb2.LoadModelResponse(ok=False, error="Укажите hf_model_id")
        if self._master_node is None:
            return cluster_pb2.LoadModelResponse(ok=False, error="Мастер не готов к загрузке модели")
        ok, err = self._master_node.load_model(request.hf_model_id.strip())
        return cluster_pb2.LoadModelResponse(ok=ok, error=err or "")

    def UnloadModel(
        self,
        request: cluster_pb2.UnloadModelRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.UnloadModelResponse:
        if self._master_node is None:
            return cluster_pb2.UnloadModelResponse(ok=False, error="Мастер не готов")
        model_id = (request.model_id or "").strip() or None
        ok, err = self._master_node.unload_model(model_id)
        return cluster_pb2.UnloadModelResponse(ok=ok, error=err or "")

    def UpdateWorkersConfig(
        self,
        request: cluster_pb2.UpdateWorkersConfigRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.UpdateWorkersConfigResponse:
        if self._master_node is None:
            return cluster_pb2.UpdateWorkersConfigResponse(ok=False, error="Мастер не готов")
        from cluster_core.common.config import WorkerConfig
        workers = [
            WorkerConfig(host=w.host or "", port=int(w.port) if w.port else 0, auth_token=w.auth_token or None)
            for w in request.workers
        ]
        workers = [w for w in workers if w.host and w.port > 0]
        try:
            ok, err = self._master_node.update_workers_config(workers)
            return cluster_pb2.UpdateWorkersConfigResponse(ok=ok, error=err or "")
        except Exception as e:
            return cluster_pb2.UpdateWorkersConfigResponse(ok=False, error=str(e))
