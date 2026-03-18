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
from pathlib import Path
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
        # streaming_chunks: храним только путь к кэшу HF и диапазон слоёв; веса подгружаются по мере надобности.
        self._stream_plans: Dict[str, Dict[str, Any]] = {}  # module_key -> {path, model_id, start, end}
        self._load_progress_lock = threading.Lock()
        self._load_progress: Dict[str, Any] | None = None  # во время InitShard (HF): stage, percent, bytes_*

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
                    logger.info("Начало скачивания с HuggingFace (с прогрессом)...")
                    path = download_repo_with_progress(
                        request.hf_model_name,
                        progress_callback=on_progress,
                    )
                    logger.info("Артефакты модели скачаны в кэш: %s", path)
                    shard_keys = list(request.shard_keys) if request.shard_keys else []
                    if shard_keys:
                        logger.info("Получены metки шарда (keys=%d). Загружаем только нужные тензоры...", len(shard_keys))
                    state_dict = load_state_dict_from_dir(path, request.hf_model_name, keys=shard_keys or None)
                finally:
                    with self._load_progress_lock:
                        self._load_progress = None

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
                    module = _try_build_layers_module(spec.model_id, state_dict)
                    if module is not None:
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

        # Убираем файлы/каталоги, из-за которых git pull выдаёт "untracked working tree files would be overwritten by merge"
        pid_file = project_root / ".chaboss_gui.pid"
        if pid_file.exists():
            try:
                pid_file.unlink()
            except OSError:
                pass
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
            return cluster_pb2.RemoteUpdateResponse(
                ok=False,
                error=f"git pull failed (code={out.returncode})",
                output=output,
            )

        if request.restart_gui:
            try:
                self._restart_gui(project_root=project_root, start_worker=bool(request.start_worker))
                output += ("\n" if output else "") + "GUI restart requested"
            except Exception as e:  # noqa: BLE001
                return cluster_pb2.RemoteUpdateResponse(ok=False, error=f"GUI restart failed: {e}", output=output)

        return cluster_pb2.RemoteUpdateResponse(ok=True, error="", output=output)

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

