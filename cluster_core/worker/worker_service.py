from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import platform
import hashlib
import psutil
import threading
import time
import sys
import os
import base64
from pathlib import Path
from typing import Any, Dict, Iterator, List

import grpc
import torch
import torch.nn as nn

from cluster_core.common.tensor_io import payload_to_tensor, tensor_to_payload
from cluster_core.common.types import GpuInfo, ResourceInfo, WorkerDescriptor, WorkerId, WorkerStatus
from cluster_core.common.chat_storage import ChatStorage, StoredChatAttachment
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc

logger = logging.getLogger("worker.service")

# Паттерны ключей по архитектурам
_LAYER_PREFIX_RE = re.compile(r"^(?:bert\.)?encoder\.layer\.(\d+)\.")
_LAYER_PREFIX_RE_GPT2 = re.compile(r"^transformer\.h\.(\d+)\.")
_LAYER_PREFIX_RE_LLAMA = re.compile(r"^model\.layers\.(\d+)\.")


def _load_state_dict_into(module: nn.Module, state_dict: dict, strict: bool = False) -> None:
    """Загружает state_dict в модуль; при возможности с assign=True (PyTorch 2+) — без копирования, пик памяти 1x."""
    try:
        module.load_state_dict(state_dict, strict=strict, assign=True)
    except TypeError:
        module.load_state_dict(state_dict, strict=strict)


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
                _load_state_dict_into(layer, sub)
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
        _load_state_dict_into(layer, sub)
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
        _load_state_dict_into(layer, sub)
        layer.eval()
        layers.append(layer)
    return nn.Sequential(*layers)


def _layer_indices_from_state_dict_qwen2(state_dict: dict) -> List[int]:
    """Индексы слоёв model.layers.i (Qwen2, та же структура что у LLaMA)."""
    return _layer_indices_from_state_dict_llama(state_dict)


def _try_build_qwen3_5_moe_layers_module(model_id: str, state_dict: dict) -> nn.Module | None:
    """
    Qwen3.5 MoE (ключи model.language_model.layers.N.*).
    GPTQ-чекпоинты (qweight/qzeros) не совместимы с обычным DecoderLayer — возвращаем None.
    """
    if not any(k.startswith("model.language_model.layers.") for k in state_dict):
        return None
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    except ImportError:
        return None
    try:
        _cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        _qc = getattr(_cfg, "quantization_config", None)
        if _qc is not None:
            _qm = getattr(_qc, "quant_method", None) or (
                _qc.get("quant_method") if isinstance(_qc, dict) else None
            )
            if str(_qm).lower() == "gptq":
                logger.warning(
                    "Qwen3.5 MoE + GPTQ: сборка nn.Module на воркере не поддерживается (ключи qweight/...). "
                    "Используйте шардирование по слоям (обновлённый мастер) и режим «модель по частям» или увеличьте файл подкачки."
                )
                return None
    except Exception:
        pass
    indices: set[int] = set()
    prefix_base = "model.language_model.layers."
    for k in state_dict:
        if not k.startswith(prefix_base):
            continue
        rest = k[len(prefix_base) :].split(".", 1)[0]
        if rest.isdigit():
            indices.add(int(rest))
    indices_l = sorted(indices)
    if not indices_l:
        return None
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None
    tc = getattr(config, "text_config", None)
    if tc is None:
        return None
    if isinstance(tc, dict):
        try:
            text_cfg = Qwen3_5MoeTextConfig(**tc)
        except Exception:
            return None
    elif isinstance(tc, Qwen3_5MoeTextConfig):
        text_cfg = tc
    else:
        return None
    layers: List[nn.Module] = []
    for idx in indices_l:
        prefix_full = f"{prefix_base}{idx}."
        sub = {k[len(prefix_full) :]: v for k, v in state_dict.items() if k.startswith(prefix_full)}
        if not sub:
            return None
        layer = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=idx)
        _load_state_dict_into(layer, sub)
        layer.eval()
        layers.append(layer)
    return nn.Sequential(*layers)


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
        _load_state_dict_into(layer, sub)
        layer.eval()
        layers.append(layer)
    return nn.Sequential(*layers)


def _try_build_layers_module(model_id: str, state_dict: dict) -> nn.Module | None:
    """
    Пробует собрать Sequential слоёв по ключам state_dict.
    Порядок: BERT -> GPT-2 -> Qwen3.5 MoE (language_model.layers) -> LLaMA -> Qwen2.
    """
    if not state_dict or not model_id:
        return None
    module = _try_build_bert_layers_module(model_id, state_dict)
    if module is not None:
        return module
    module = _try_build_gpt2_layers_module(model_id, state_dict)
    if module is not None:
        return module
    module = _try_build_qwen3_5_moe_layers_module(model_id, state_dict)
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

    def __init__(
        self,
        host: str,
        port: int,
        auth_token: str | None = None,
        log_buffer: Any = None,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._auth_token_fingerprint = (
            hashlib.sha256(auth_token.encode("utf-8")).hexdigest()[:12] if auth_token else ""
        )
        self._resources = _detect_resources()
        self._shards: Dict[str, Dict[str, Any]] = {}  # shard_id -> state_dict chunk
        self._shard_modules: Dict[str, nn.Module] = {}  # shard_id -> nn.Module (BERT layers и т.д.)
        self._stream_plans: Dict[str, Dict[str, Any]] = {}  # module_key -> {path, model_id, start, end}
        self._load_progress_lock = threading.Lock()
        self._load_progress: Dict[str, Any] | None = None
        self._log_buffer = log_buffer  # для GetWorkerLogs (буфер из run_worker)

        # Чат: история и вложения в ./cluster_shared/
        project_root = Path(__file__).resolve().parents[2]
        shared_root = project_root / "cluster_shared"
        self._chat_storage = ChatStorage(shared_root, cache_messages=500)

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

    def GetLoadProgress(
        self,
        request: cluster_pb2.WorkerId,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.GetLoadProgressResponse:
        with self._load_progress_lock:
            p = self._load_progress
        if not p:
            return cluster_pb2.GetLoadProgressResponse(progress=cluster_pb2.LoadProgress(stage="", percent=0))
        return cluster_pb2.GetLoadProgressResponse(
            progress=cluster_pb2.LoadProgress(
                stage=p.get("stage", "download"),
                percent=int(p.get("percent", 0)),
                bytes_downloaded=int(p.get("bytes_downloaded", 0)),
                bytes_total=int(p.get("bytes_total", 0)),
                current_file=p.get("current_file", "") or "",
            )
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
            "InitShard: получена команда — worker=%s:%s model_id=%s shard_id=%s source=%s shard_keys_count=%d",
            self._host,
            self._port,
            spec.model_id, spec.shard_id, request.weight_source,
            len(request.shard_keys) if request.shard_keys is not None else 0,
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
                if state_dict and spec.model_id:
                    logger.info("Сборка модуля слоёв для model_id=%s...", spec.model_id)
                    module = _try_build_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
                        self._shards[shard_id] = {}
                        del state_dict
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info("Модуль слоёв собран успешно")
                    else:
                        self._shards[shard_id] = state_dict
                        logger.info("Модуль слоёв не собран (будут только веса)")
                else:
                    self._shards[shard_id] = state_dict
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
                num_keys = len(state_dict)
                if state_dict and spec.model_id:
                    module = _try_build_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        self._shard_modules[shard_id] = module
                        self._shards[shard_id] = {}
                        del state_dict
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        self._shards[shard_id] = state_dict
                else:
                    self._shards[shard_id] = state_dict
                elapsed = time.perf_counter() - t0
                logger.info("Ответ мастеру: шард из shared_path загружен, ключей=%d, время=%.2f с", num_keys, elapsed)
            elif request.weight_source == "hf" and request.hf_model_name:
                logger.info("Получена команда скачать модель с HF: %s", request.hf_model_name)
                try:
                    from cluster_core.common.hf_download import download_repo_with_progress, load_state_dict_from_dir
                except ImportError as e:
                    return cluster_pb2.InitShardResponse(ok=False, error=f"hf_download недоступен: {e}")

                shard_keys = list(request.shard_keys) if request.shard_keys else []
                def on_progress(percent: float, bytes_done: int, bytes_total: int, current_file: str) -> None:
                    # Чтобы не спамить лог, выводим только при заметных изменениях.
                    # (Именно логов часто не хватает, когда процесс падает после прогресса.)
                    p_int = min(100, int(percent or 0))
                    do_log = False
                    if not hasattr(on_progress, "_last_percent"):
                        setattr(on_progress, "_last_percent", -1)
                        setattr(on_progress, "_last_file", "")
                    last_percent = getattr(on_progress, "_last_percent")
                    last_file = getattr(on_progress, "_last_file")
                    if p_int >= 100 or p_int - last_percent >= 10 or current_file != last_file:
                        do_log = True
                        setattr(on_progress, "_last_percent", p_int)
                        setattr(on_progress, "_last_file", current_file or "")

                    with self._load_progress_lock:
                        self._load_progress = {
                            "stage": "download",
                            "percent": p_int,
                            "bytes_downloaded": bytes_done,
                            "bytes_total": bytes_total,
                            "current_file": current_file,
                        }
                    if do_log:
                        logger.info(
                            "InitShard[%s]: hf download progress percent=%d file=%s bytes=%d/%d",
                            shard_id,
                            p_int,
                            current_file or "",
                            bytes_done,
                            bytes_total,
                        )

                try:
                    with self._load_progress_lock:
                        self._load_progress = {"stage": "download", "percent": 0, "bytes_downloaded": 0, "bytes_total": 0, "current_file": ""}
                    logger.info(
                        "InitShard[%s]: start hf_download repo=%s keys_filter=%s (keys=%d)",
                        shard_id,
                        request.hf_model_name,
                        "ON" if shard_keys else "OFF",
                        len(shard_keys),
                    )
                    path = download_repo_with_progress(
                        request.hf_model_name,
                        progress_callback=on_progress,
                    )
                    logger.info("InitShard[%s]: hf artifacts downloaded to: %s", shard_id, path)
                    if shard_keys:
                        logger.info("InitShard[%s]: loading state_dict with filtered keys=%d", shard_id, len(shard_keys))
                    else:
                        logger.info("InitShard[%s]: loading state_dict without keys filter (may be heavy)", shard_id)
                    state_dict = load_state_dict_from_dir(path, request.hf_model_name, keys=shard_keys or None)
                finally:
                    with self._load_progress_lock:
                        self._load_progress = None

                logger.info("InitShard[%s]: state_dict loaded tensors=%d", shard_id, len(state_dict) if isinstance(state_dict, dict) else -1)
                logger.info("state_dict загружен из кэша HF, ключей=%d", len(state_dict))

                # На больших моделях важно не держать полный state_dict в памяти.
                # load_state_dict_from_dir(keys=...) уже загружает только нужные ключи (для safetensors).

                approx_bytes = 0
                try:
                    approx_bytes = sum(
                        int(v.numel()) * int(v.element_size())
                        for v in state_dict.values()
                        if hasattr(v, "numel") and hasattr(v, "element_size")
                    )
                except Exception:
                    approx_bytes = 0

                num_keys = len(state_dict)
                if state_dict and spec.model_id:
                    logger.info("InitShard[%s]: trying build layers module for model_id=%s", shard_id, spec.model_id)
                    module = _try_build_layers_module(spec.model_id, state_dict)
                    if module is not None:
                        logger.info("InitShard[%s]: layers module built type=%s", shard_id, type(module).__name__)
                        self._shard_modules[shard_id] = module
                        # Не держим state_dict в _shards — модуль уже содержит веса.
                        # Иначе пик памяти 2x (state_dict + модуль) и на Windows срабатывает os error 1455.
                        self._shards[shard_id] = {}
                        del state_dict
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        logger.info("InitShard[%s]: layers module NOT built; keeping raw weights keys=%d", shard_id, num_keys)
                        self._shards[shard_id] = state_dict
                else:
                    self._shards[shard_id] = state_dict
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Ответ мастеру: HF->шард загружен на воркере, ключей=%d, ~%.1f МБ, время=%.2f с",
                    num_keys,
                    approx_bytes / (1024 * 1024) if approx_bytes else 0.0,
                    elapsed,
                )
                logger.info("InitShard[%s]: finished ok=%s elapsed=%.2fs", shard_id, True, elapsed)
            elif request.weight_source == "hf_stream" and request.hf_model_name:
                logger.info("InitShard streaming: HF=%s shard_id=%s", request.hf_model_name, spec.shard_id)
                try:
                    from cluster_core.common.hf_download import download_repo_with_progress
                except ImportError as e:
                    return cluster_pb2.InitShardResponse(ok=False, error=f"hf_download недоступен: {e}")

                # shard_id ожидается как "start-end" (end не включительно)
                try:
                    start_s, end_s = (spec.shard_id or "").split("-", 1)
                    start_i, end_i = int(start_s), int(end_s)
                    if end_i <= start_i:
                        raise ValueError("empty range")
                except Exception as e:
                    return cluster_pb2.InitShardResponse(ok=False, error=f"Некорректный shard_id для streaming: {spec.shard_id} ({e})")

                def on_progress(percent: float, bytes_done: int, bytes_total: int, current_file: str) -> None:
                    with self._load_progress_lock:
                        self._load_progress = {
                            "stage": "download",
                            "percent": min(100, int(percent)),
                            "bytes_downloaded": bytes_done,
                            "bytes_total": bytes_total,
                            "current_file": current_file,
                        }

                try:
                    with self._load_progress_lock:
                        self._load_progress = {"stage": "download", "percent": 0, "bytes_downloaded": 0, "bytes_total": 0, "current_file": ""}
                    path = download_repo_with_progress(request.hf_model_name, progress_callback=on_progress)
                finally:
                    with self._load_progress_lock:
                        self._load_progress = None

                module_key = f"{spec.model_id}:{spec.shard_id}"
                self._stream_plans[module_key] = {
                    "path": str(path),
                    "model_id": request.hf_model_name,
                    "start": start_i,
                    "end": end_i,
                }
                elapsed = time.perf_counter() - t0
                logger.info("InitShard streaming: план сохранён, слои [%d..%d), время=%.2f с", start_i, end_i, elapsed)
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
        try:
            for ping in request_iterator:
                yield cluster_pb2.HealthPong(
                    id=self._to_worker_id(),
                    nonce=ping.nonce,
                    status=cluster_pb2.WORKER_STATUS_ONLINE,
                )
        except Exception as e:
            logger.warning("HealthStream обрыв (воркер): %s", e)

    def RemoteUpdate(
        self,
        request: cluster_pb2.RemoteUpdateRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.RemoteUpdateResponse:
        """
        Удалённое обновление воркера: выполняет git pull в корне проекта.
        Опционально перезапускает GUI (если он запущен) с флагом --start-worker.
        """
        project_root = Path(__file__).resolve().parents[2]  # .../chaboss_cluster
        git_remote = (request.git_remote or "origin").strip() or "origin"
        git_branch = (request.git_branch or "").strip()

        # Убираем каталоги, из-за которых git pull может падать на untracked-конфликтах.
        # PID-файл GUI НЕ трогаем: он нужен для корректного авто-перезапуска GUI после обновления.
        for pycache in project_root.rglob("__pycache__"):
            if pycache.is_dir():
                try:
                    shutil.rmtree(pycache, ignore_errors=True)
                except OSError:
                    pass

        cmd = ["git", "pull", "--rebase", "--autostash", git_remote]
        if git_branch:
            cmd.append(git_branch)

        try:
            out = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=180,
            )
        except Exception as e:  # noqa: BLE001
            return cluster_pb2.RemoteUpdateResponse(ok=False, error=str(e), output="")

        output = ""
        if out.stdout:
            output += out.stdout
        if out.stderr:
            output += ("\n" if output else "") + out.stderr

        if out.returncode != 0:
            err = f"git pull failed (code={out.returncode})"
            low = output.lower()
            if any(
                s in low
                for s in (
                    "unmerged files",
                    "unresolved conflict",
                    "merge conflict",
                    "you need to resolve",
                    "fix them up in the work tree",
                    "rebase conflict",
                )
            ):
                err += (
                    " — конфликт слияния/rebase в репозитории на этой машине. "
                    "Зайдите в каталог проекта на воркере, выполните `git status`, "
                    "разрешите конфликты, затем `git add` нужные файлы и "
                    "`git commit` (или отмените операцию: `git merge --abort` / `git rebase --abort`), "
                    "после чего снова `git pull`. Затем перезапустите воркер."
                )
            return cluster_pb2.RemoteUpdateResponse(
                ok=False,
                error=err,
                output=output,
            )

        if request.restart_gui:
            try:
                self._restart_gui(project_root=project_root, start_worker=bool(request.start_worker))
                output += ("\n" if output else "") + "GUI restart requested"
            except Exception as e:  # noqa: BLE001
                return cluster_pb2.RemoteUpdateResponse(ok=False, error=f"GUI restart failed: {e}", output=output)

        return cluster_pb2.RemoteUpdateResponse(ok=True, error="", output=output)

    def GetWorkerLogs(
        self,
        request: cluster_pb2.GetWorkerLogsRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.GetWorkerLogsResponse:
        """Отдаёт последние строки из буфера логов воркера (то же, что пишется в logs/worker.log)."""
        since = max(0, getattr(request, "since_index", 0) or 0)
        if self._log_buffer is None:
            return cluster_pb2.GetWorkerLogsResponse(lines=[], next_index=0)
        lines, next_index = self._log_buffer.get_since(since)
        return cluster_pb2.GetWorkerLogsResponse(lines=lines, next_index=next_index)

    def ReceiveChatMessage(
        self,
        request_iterator: Iterator[cluster_pb2.ChatPostChunk],
        context: grpc.ServicerContext,
    ) -> cluster_pb2.ReceiveChatMessageResponse:
        """
        Client-streaming: мастер отправляет header + чанки вложений.
        Идемпотентность: если message_id уже сохранён — повторно не пишем.
        """
        header: cluster_pb2.ChatPostHeader | None = None
        attachments_meta: list[cluster_pb2.ChatPostAttachmentMeta] = []

        tmp_fds: Dict[str, Any] = {}
        bytes_received: Dict[str, int] = {}
        is_last_seen: Dict[str, bool] = {}
        skip_store = False

        try:
            for chunk in request_iterator:
                which = chunk.WhichOneof("payload")
                if which == "header":
                    header = chunk.header
                    attachments_meta = list(header.attachments or [])
                    if not header.message_id:
                        return cluster_pb2.ReceiveChatMessageResponse(ok=False, error="message_id обязателен.", message_id="")
                    # Идемпотентность можно включать после header.
                    skip_store = self._chat_storage.has_received_message(header.message_id)

                    if not skip_store:
                        # открываем tmp-файлы под вложения
                        for a in attachments_meta:
                            if a.size > 20 * 1024 * 1024:
                                return cluster_pb2.ReceiveChatMessageResponse(ok=False, error="Слишком большой файл.", message_id=header.message_id)
                            orig_path = self._chat_storage.attachment_original_path(a.attachment_id)
                            tmp_path = orig_path.with_suffix(".tmp")
                            tmp_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp_fds[a.attachment_id] = tmp_path.open("wb")
                            bytes_received[a.attachment_id] = 0
                            is_last_seen[a.attachment_id] = False

                elif which == "attachment_chunk":
                    if header is None:
                        return cluster_pb2.ReceiveChatMessageResponse(ok=False, error="Сначала должен прийти header.", message_id="")
                    ac = chunk.attachment_chunk
                    aid = str(ac.attachment_id or "").strip()
                    if skip_store:
                        continue
                    if not aid or aid not in tmp_fds:
                        continue
                    data = ac.data or b""
                    tmp_fds[aid].write(data)
                    bytes_received[aid] += len(data)
                    is_last_seen[aid] = bool(is_last_seen.get(aid, False) or ac.is_last)

                else:
                    continue
        except Exception as e:  # noqa: BLE001
            return cluster_pb2.ReceiveChatMessageResponse(ok=False, error=str(e), message_id=header.message_id if header else "")

        if header is None:
            return cluster_pb2.ReceiveChatMessageResponse(ok=False, error="Header не получен.", message_id="")

        try:
            if skip_store:
                return cluster_pb2.ReceiveChatMessageResponse(ok=True, error="", message_id=header.message_id)

            # close + rename temp files
            for aid, fd in tmp_fds.items():
                fd.flush()
                fd.close()
                orig_path = self._chat_storage.attachment_original_path(aid)
                tmp_path = orig_path.with_suffix(".tmp")
                tmp_path.replace(orig_path)

            for a in attachments_meta:
                if not is_last_seen.get(a.attachment_id, False):
                    return cluster_pb2.ReceiveChatMessageResponse(ok=False, error=f"Не получены все чанки для {a.filename}", message_id=header.message_id)

            # thumbnails + record attachments
            attachments_for_record: list[StoredChatAttachment] = []
            for a in attachments_meta:
                thumb_bytes = bytes(a.thumbnail_jpeg or b"")
                if a.is_image and thumb_bytes:
                    self._chat_storage.write_thumbnail_bytes(a.attachment_id, thumb_bytes)
                    thumb_b64 = base64.b64encode(thumb_bytes).decode("ascii")
                else:
                    thumb_b64 = None
                attachments_for_record.append(
                    StoredChatAttachment(
                        attachment_id=a.attachment_id,
                        filename=a.filename,
                        mime_type=a.mime_type,
                        is_image=bool(a.is_image),
                        size=int(a.size or 0),
                        thumbnail_jpeg_b64=thumb_b64,
                    )
                )

            self._chat_storage.append_message(
                message_id=header.message_id,
                timestamp_ms=int(header.timestamp_ms or 0),
                channel_id=header.channel_id,
                sender=header.sender or "",
                text=header.text or "",
                attachments=attachments_for_record,
            )
            self._chat_storage.save_received_marker(header.message_id)
            return cluster_pb2.ReceiveChatMessageResponse(ok=True, error="", message_id=header.message_id)
        except Exception as e:  # noqa: BLE001
            return cluster_pb2.ReceiveChatMessageResponse(ok=False, error=str(e), message_id=header.message_id)

    def _restart_gui(self, project_root: Path, start_worker: bool) -> None:
        """
        Best-effort: если найден PID-файл GUI — останавливает процесс и запускает новый.
        Если GUI не запущен (PID-файла нет) — ничего не делает.
        Запуск: python -m ui.main_window [--start-worker]
        """
        pid_file = project_root / ".chaboss_gui.pid"
        if not pid_file.exists():
            return
        try:
            pid_txt = pid_file.read_text(encoding="utf-8").strip()
            pid = int(pid_txt)
        except Exception:
            pid = 0
        if pid > 0:
            try:
                psutil.Process(pid).terminate()
            except Exception:
                pass

        # В headless-окружении GUI не стартуем.
        if platform.system().lower() == "linux":
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                return

        args = [sys.executable, "-m", "ui.main_window"]
        if start_worker:
            args.append("--start-worker")
        subprocess.Popen(
            args,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
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
                elif module_key and module_key in self._stream_plans:
                    out_tensor = self._run_streaming_layers(self._stream_plans[module_key], tensor)
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

    def _run_streaming_layers(self, plan: Dict[str, Any], tensor: "torch.Tensor") -> "torch.Tensor":
        """
        Прогоняет tensor через слои [start..end) для модели, подгружая веса слоя из safetensors по мере надобности.
        Не держит большой state_dict в RAM — подходит для слабых воркеров.
        """
        import pathlib
        from safetensors import safe_open
        from transformers import AutoConfig

        model_id = str(plan.get("model_id") or "")
        start = int(plan.get("start", 0))
        end = int(plan.get("end", 0))
        base_path = pathlib.Path(str(plan.get("path") or ""))
        st_files = sorted(base_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"Нет .safetensors в {base_path}")

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        mt = getattr(cfg, "model_type", "") or ""
        use_lm_prefix = mt == "qwen3_5_moe" or type(cfg).__name__ == "Qwen3_5MoeConfig"

        def layer_key_prefix(layer_idx: int) -> str:
            if use_lm_prefix:
                return f"model.language_model.layers.{layer_idx}."
            return f"model.layers.{layer_idx}."

        def load_layer_state(layer_idx: int) -> Dict[str, Any]:
            prefix = layer_key_prefix(layer_idx)
            sd: Dict[str, Any] = {}
            for sf in st_files:
                with safe_open(str(sf), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k.startswith(prefix):
                            sd[k[len(prefix) :]] = f.get_tensor(k)
            return sd

        def build_layer(layer_idx: int, layer_sd: Dict[str, Any]) -> nn.Module:
            if use_lm_prefix:
                try:
                    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
                    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
                except ImportError as e:
                    raise RuntimeError(f"Нужен transformers с Qwen3.5 MoE: {e}") from e
                tc = getattr(cfg, "text_config", None)
                if isinstance(tc, dict):
                    text_cfg = Qwen3_5MoeTextConfig(**tc)
                elif isinstance(tc, Qwen3_5MoeTextConfig):
                    text_cfg = tc
                else:
                    raise RuntimeError("Qwen3.5 MoE: нет text_config")
                layer = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=layer_idx)
                _load_state_dict_into(layer, layer_sd)
                layer.eval()
                return layer
            layer: nn.Module | None = None
            try:
                from transformers import Qwen2DecoderLayer, Qwen2Config
                if isinstance(cfg, Qwen2Config):
                    layer = Qwen2DecoderLayer(cfg)
            except Exception:
                layer = None
            if layer is None:
                try:
                    from transformers import LlamaDecoderLayer, LlamaConfig
                    if isinstance(cfg, LlamaConfig):
                        layer = LlamaDecoderLayer(cfg)
                except Exception:
                    layer = None
            if layer is None:
                from transformers import LlamaDecoderLayer
                layer = LlamaDecoderLayer(cfg)
            _load_state_dict_into(layer, layer_sd)
            layer.eval()
            return layer

        with torch.no_grad():
            cur = tensor
            if cur.dim() == 2:
                cur = cur.unsqueeze(0)
            for idx in range(start, end):
                layer_sd = load_layer_state(idx)
                if not layer_sd:
                    raise KeyError(f"Не найдены ключи слоя {layer_key_prefix(idx)}* в safetensors")
                layer = build_layer(idx, layer_sd)
                out = layer(cur)
                if isinstance(out, tuple):
                    out = out[0]
                cur = out
                del layer
                del layer_sd
            return cur

