from __future__ import annotations

import io
import logging
import re
import subprocess
import platform
import hashlib
import psutil
import time
from typing import Any, Dict, Iterator, List

import grpc
import torch
import torch.nn as nn

from cluster_core.common.tensor_io import payload_to_tensor, tensor_to_payload
from cluster_core.common.types import GpuInfo, ResourceInfo, WorkerDescriptor, WorkerId, WorkerStatus
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc

logger = logging.getLogger("worker.service")

# Паттерны ключей по архитектурам
_LAYER_PREFIX_RE = re.compile(r"^(?:bert\.)?encoder\.layer\.(\d+)\.")
_LAYER_PREFIX_RE_GPT2 = re.compile(r"^transformer\.h\.(\d+)\.")
_LAYER_PREFIX_RE_LLAMA = re.compile(r"^model\.layers\.(\d+)\.")


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


def _layer_indices_from_state_dict_gpt2(state_dict: dict) -> List[int]:
    """Индексы слоёв transformer.h.i (GPT-2)."""
    indices = set()
    for k in state_dict:
        m = _LAYER_PREFIX_RE_GPT2.match(k)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _try_build_gpt2_layers_module(model_id: str, state_dict: dict) -> nn.Module | None:
    """
    Собирает nn.Module из state_dict для GPT-2 (transformer.h.i).
    Возвращает Sequential из GPT2Block или None при ошибке.
    """
    try:
        from transformers import GPT2Config, GPT2Block
    except ImportError:
        return None
    indices = _layer_indices_from_state_dict_gpt2(state_dict)
    if not indices:
        return None
    try:
        config = GPT2Config.from_pretrained(model_id)
    except Exception:
        return None
    layers: List[nn.Module] = []
    prefix = "transformer.h."
    for idx in indices:
        prefix_full = f"{prefix}{idx}."
        sub = {k[len(prefix_full):]: v for k, v in state_dict.items() if k.startswith(prefix_full)}
        if not sub:
            return None
        layer = GPT2Block(config)
        layer.load_state_dict(sub, strict=False)
        layer.eval()
        layers.append(layer)
    return nn.Sequential(*layers)


def _layer_indices_from_state_dict_llama(state_dict: dict) -> List[int]:
    """Индексы слоёв model.layers.i (LLaMA)."""
    indices = set()
    for k in state_dict:
        m = _LAYER_PREFIX_RE_LLAMA.match(k)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _try_build_llama_layers_module(model_id: str, state_dict: dict) -> nn.Module | None:
    """
    Собирает nn.Module из state_dict для LLaMA (model.layers.i).
    Возвращает Sequential из LlamaDecoderLayer или None при ошибке.
    """
    try:
        from transformers import LlamaConfig, LlamaDecoderLayer
    except ImportError:
        try:
            from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer
        except ImportError:
            return None
    indices = _layer_indices_from_state_dict_llama(state_dict)
    if not indices:
        return None
    try:
        config = LlamaConfig.from_pretrained(model_id)
    except Exception:
        return None
    layers: List[nn.Module] = []
    prefix = "model.layers."
    for idx in indices:
        prefix_full = f"{prefix}{idx}."
        sub = {k[len(prefix_full):]: v for k, v in state_dict.items() if k.startswith(prefix_full)}
        if not sub:
            return None
        layer = LlamaDecoderLayer(config)
        layer.load_state_dict(sub, strict=False)
        layer.eval()
        layers.append(layer)
    return nn.Sequential(*layers)


def _layer_indices_from_state_dict_qwen2(state_dict: dict) -> List[int]:
    """Индексы слоёв model.layers.i (Qwen2, та же структура что у LLaMA)."""
    return _layer_indices_from_state_dict_llama(state_dict)


def _try_build_qwen2_layers_module(model_id: str, state_dict: dict) -> nn.Module | None:
    """
    Собирает nn.Module из state_dict для Qwen2 (model.layers.i).
    Возвращает Sequential из Qwen2DecoderLayer или None при ошибке.
    """
    try:
        from transformers import Qwen2Config, Qwen2DecoderLayer
    except ImportError:
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2DecoderLayer
        except ImportError:
            return None
    indices = _layer_indices_from_state_dict_qwen2(state_dict)
    if not indices:
        return None
    try:
        config = Qwen2Config.from_pretrained(model_id)
    except Exception:
        return None
    layers: List[nn.Module] = []
    prefix = "model.layers."
    for idx in indices:
        prefix_full = f"{prefix}{idx}."
        sub = {k[len(prefix_full):]: v for k, v in state_dict.items() if k.startswith(prefix_full)}
        if not sub:
            return None
        layer = Qwen2DecoderLayer(config)
        layer.load_state_dict(sub, strict=False)
        layer.eval()
        layers.append(layer)
    return nn.Sequential(*layers)


def _try_build_layers_module(model_id: str, state_dict: dict) -> nn.Module | None:
    """
    Пробует собрать Sequential слоёв по ключам state_dict.
    Порядок: BERT -> GPT-2 -> LLaMA -> Qwen2.
    """
    if not state_dict or not model_id:
        return None
    module = _try_build_bert_layers_module(model_id, state_dict)
    if module is not None:
        return module
    module = _try_build_gpt2_layers_module(model_id, state_dict)
    if module is not None:
        return module
    module = _try_build_llama_layers_module(model_id, state_dict)
    if module is not None:
        return module
    return _try_build_qwen2_layers_module(model_id, state_dict)


def _detect_gpus_nvidia_smi() -> list[GpuInfo]:
    """Резервное определение GPU через nvidia-smi (если PyTorch без CUDA)."""
    gpus: list[GpuInfo] = []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return gpus
        for idx, line in enumerate(out.stdout.strip().split("\n")):
            parts = [p.strip() for p in line.split(",", 1)]
            name = parts[0] if parts else f"GPU {idx}"
            total_mb = 0
            if len(parts) > 1:
                try:
                    total_mb = int(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass
            gpus.append(
                GpuInfo(
                    index=idx,
                    name=name,
                    total_vram_mb=total_mb,
                    compute_capability=None,
                    backend="cuda",
                )
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):  # noqa: BLE001, S110
        pass
    return gpus


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
    if not gpus:
        gpus = _detect_gpus_nvidia_smi()

    return ResourceInfo(
        cpu_cores=cpu_cores,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        gpus=gpus,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.version.cuda is not None else None,
        rocm_version=None,
        os_name=platform.system().lower(),
        os_version=f"{platform.release()} {platform.version()}".strip(),
    )


class WorkerService(cluster_pb2_grpc.WorkerServiceServicer):
    """
    gRPC‑сервис воркера: GetStatus, InitShard (сборка BERT‑слоёв при возможности),
    HealthStream, RunStage (forward по модулю или identity).
    """

    def __init__(self, host: str, port: int, auth_token: str | None = None) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._auth_token_fingerprint = (
            hashlib.sha256(auth_token.encode("utf-8")).hexdigest()[:12] if auth_token else ""
        )
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
            os_name=self._resources.os_name or "",
            os_version=self._resources.os_version or "",
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
            token_fingerprint=self._auth_token_fingerprint,
            token_status="UNKNOWN",
        )

    def InitShard(
        self,
        request: cluster_pb2.InitShardRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.InitShardResponse:
        t0 = time.perf_counter()
        spec = request.spec
        shard_id = f"{spec.model_id}:{spec.shard_id}"
        logger.info(
            "InitShard: получена команда — model_id=%s, shard_id=%s, source=%s",
            spec.model_id, spec.shard_id, request.weight_source,
        )
        try:
            if request.weight_source == "inline_blob":
                blob_size = len(request.inline_blob) if request.inline_blob else 0
                logger.info("Получен inline_blob от мастера, размер=%d байт (метка шарда: %s)", blob_size, shard_id)
                state_dict = {}
                if request.inline_blob:
                    logger.info("Загрузка state_dict из blob...")
                    state_dict = torch.load(
                        io.BytesIO(request.inline_blob),
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not isinstance(state_dict, dict):
                        return cluster_pb2.InitShardResponse(ok=False, error="Ожидался state_dict (dict)")
                    logger.info("state_dict загружен из blob, ключей=%d", len(state_dict))
                self._shards[shard_id] = state_dict
                if state_dict and spec.model_id:
                    logger.info("Сборка модуля слоёв для model_id=%s...", spec.model_id)
                    module = _try_build_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
                        logger.info("Модуль слоёв собран успешно")
                    else:
                        logger.info("Модуль слоёв не собран (будут только веса)")
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Ответ мастеру: чанк загружен успешно, размер=%d байт, время=%.2f с",
                    blob_size, elapsed,
                )
            elif request.weight_source == "shared_path" and request.shared_path:
                logger.info("Загрузка шарда из shared_path: %s", request.shared_path)
                state_dict = torch.load(
                    request.shared_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(state_dict, dict):
                    return cluster_pb2.InitShardResponse(ok=False, error="Ожидался state_dict (dict)")
                self._shards[shard_id] = state_dict
                if state_dict and spec.model_id:
                    module = _try_build_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
                elapsed = time.perf_counter() - t0
                logger.info("Ответ мастеру: шард из shared_path загружен, ключей=%d, время=%.2f с", len(state_dict), elapsed)
            elif request.weight_source == "hf" and request.hf_model_name:
                logger.info("Получена команда скачать модель с HF: %s", request.hf_model_name)
                from huggingface_hub import hf_hub_download
                logger.info("Начало скачивания с HuggingFace...")
                path = hf_hub_download(repo_id=request.hf_model_name, filename="pytorch_model.bin")
                logger.info("Скачивание с HF завершено: %s", path)
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(state_dict, dict):
                    state_dict = getattr(state_dict, "state_dict", lambda: {})()
                logger.info("state_dict загружен с диска, ключей=%d", len(state_dict))

                # Если мастер прислал список ключей шарда — оставляем только их.
                shard_keys = list(request.shard_keys) if request.shard_keys else []
                if shard_keys:
                    logger.info("Получены метки шарда (keys=%d). Фильтрация state_dict...", len(shard_keys))
                    filtered = {k: state_dict[k] for k in shard_keys if k in state_dict}
                    missing = len(shard_keys) - len(filtered)
                    state_dict = filtered
                    logger.info("Фильтрация завершена: keys=%d (missing=%d)", len(state_dict), missing)

                approx_bytes = 0
                try:
                    approx_bytes = sum(
                        int(v.numel()) * int(v.element_size())
                        for v in state_dict.values()
                        if hasattr(v, "numel") and hasattr(v, "element_size")
                    )
                except Exception:
                    approx_bytes = 0

                self._shards[shard_id] = state_dict
                if state_dict and spec.model_id:
                    module = _try_build_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Ответ мастеру: HF->шард загружен на воркере, ключей=%d, ~%.1f МБ, время=%.2f с",
                    len(state_dict),
                    approx_bytes / (1024 * 1024) if approx_bytes else 0.0,
                    elapsed,
                )
            else:
                return cluster_pb2.InitShardResponse(ok=False, error="Не задан источник весов")
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.exception("InitShard failed за %.2f с: %s", elapsed, e)
            return cluster_pb2.InitShardResponse(ok=False, error=str(e))
        return cluster_pb2.InitShardResponse(ok=True, error="")

    def UnloadShard(
        self,
        request: cluster_pb2.UnloadShardRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.UnloadShardResponse:
        model_id = (request.model_id or "").strip()
        shard_id = (request.shard_id or "").strip()
        if not model_id and not shard_id:
            keys_to_drop = list(self._shards.keys())
        elif model_id and shard_id:
            keys_to_drop = [f"{model_id}:{shard_id}"] if f"{model_id}:{shard_id}" in self._shards else []
        elif model_id:
            keys_to_drop = [k for k in self._shards if k.startswith(f"{model_id}:")]
        else:
            keys_to_drop = [k for k in self._shards if k.endswith(f":{shard_id}")]
        for key in keys_to_drop:
            self._shards.pop(key, None)
            self._shard_modules.pop(key, None)
        if keys_to_drop and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cluster_pb2.UnloadShardResponse(ok=True, error="")

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

