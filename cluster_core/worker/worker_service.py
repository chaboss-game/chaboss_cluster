from __future__ import annotations

import io
import re
import psutil
from typing import Any, Dict, Iterator, List

import grpc
import torch
import torch.nn as nn

from cluster_core.common.tensor_io import payload_to_tensor, tensor_to_payload
from cluster_core.common.types import GpuInfo, ResourceInfo, WorkerDescriptor, WorkerId, WorkerStatus
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc

# Паттерн ключей encoder.layer.N или bert.encoder.layer.N
_LAYER_PREFIX_RE = re.compile(r"^(?:bert\.)?encoder\.layer\.(\d+)\.")


def _layer_indices_from_state_dict(state_dict: dict) -> List[int]:
    """Возвращает отсортированный список индексов слоёв по ключам state_dict."""
    indices = set()
    for k in state_dict:
        m = _LAYER_PREFIX_RE.match(k)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _try_build_bert_layers_module(model_id: str, state_dict: dict) -> nn.Module | None:
    """
    Собирает nn.Module из state_dict для BERT-слоёв (encoder.layer.i).
    Возвращает Sequential из BertLayer или None при ошибке.
    """
    try:
        from transformers import BertConfig, BertLayer
    except ImportError:
        return None
    indices = _layer_indices_from_state_dict(state_dict)
    if not indices:
        return None
    config = BertConfig.from_pretrained(model_id)
    layers: List[nn.Module] = []
    prefix_candidates = ("encoder.layer.", "bert.encoder.layer.")
    for idx in indices:
        layer = BertLayer(config)
        loaded = False
        for prefix in prefix_candidates:
            prefix_full = f"{prefix}{idx}."
            sub = {k[len(prefix_full) :]: v for k, v in state_dict.items() if k.startswith(prefix_full)}
            if sub:
                layer.load_state_dict(sub, strict=False)
                loaded = True
                break
        if not loaded:
            return None
        layer.eval()
        layers.append(layer)
    return nn.Sequential(*layers)


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
    gRPC‑сервис воркера: GetStatus, InitShard (сборка BERT‑слоёв при возможности),
    HealthStream, RunStage (forward по модулю или identity).
    """

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._resources = _detect_resources()
        self._shards: Dict[str, Dict[str, Any]] = {}  # shard_id -> state_dict chunk
        self._shard_modules: Dict[str, nn.Module] = {}  # shard_id -> nn.Module (BERT layers и т.д.)

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
        spec = request.spec
        shard_id = f"{spec.model_id}:{spec.shard_id}"
        try:
            if request.weight_source == "inline_blob":
                state_dict = {}
                if request.inline_blob:
                    state_dict = torch.load(
                        io.BytesIO(request.inline_blob),
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not isinstance(state_dict, dict):
                        return cluster_pb2.InitShardResponse(ok=False, error="Ожидался state_dict (dict)")
                self._shards[shard_id] = state_dict
                # Попытка собрать модуль для forward (BERT-слои)
                if state_dict and spec.model_id:
                    module = _try_build_bert_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
            elif request.weight_source == "shared_path" and request.shared_path:
                state_dict = torch.load(
                    request.shared_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(state_dict, dict):
                    return cluster_pb2.InitShardResponse(ok=False, error="Ожидался state_dict (dict)")
                self._shards[shard_id] = state_dict
                if state_dict and spec.model_id:
                    module = _try_build_bert_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
            elif request.weight_source == "hf" and request.hf_model_name:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=request.hf_model_name, filename="pytorch_model.bin")
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(state_dict, dict):
                    state_dict = getattr(state_dict, "state_dict", lambda: {})()
                self._shards[shard_id] = state_dict
                if state_dict and spec.model_id:
                    module = _try_build_bert_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
            else:
                return cluster_pb2.InitShardResponse(ok=False, error="Не задан источник весов")
        except Exception as e:  # noqa: BLE001
            return cluster_pb2.InitShardResponse(ok=False, error=str(e))
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
        for req in request_iterator:
            if not req.tensor or not req.tensor.data:
                yield cluster_pb2.StageResponse(
                    request_id=req.request_id,
                    tensor=req.tensor,
                    is_last=req.is_last,
                    error="пустой тензор",
                )
                continue
            try:
                tensor = payload_to_tensor(req.tensor, device="cpu")
                module_key = f"{req.model_id}:{req.shard_id}" if (req.model_id and req.shard_id) else None
                if module_key and module_key in self._shard_modules:
                    mod = self._shard_modules[module_key]
                    with torch.no_grad():
                        if tensor.dim() == 2:
                            tensor = tensor.unsqueeze(0)
                        out_tensor = mod(tensor)
                        if isinstance(out_tensor, tuple):
                            out_tensor = out_tensor[0]
                else:
                    out_tensor = tensor
                out_payload = tensor_to_payload(out_tensor)
            except Exception as e:
                yield cluster_pb2.StageResponse(
                    request_id=req.request_id,
                    tensor=cluster_pb2.TensorPayload(),
                    is_last=req.is_last,
                    error=str(e),
                )
                continue
            yield cluster_pb2.StageResponse(
                request_id=req.request_id,
                tensor=out_payload,
                is_last=req.is_last,
                error="",
            )

