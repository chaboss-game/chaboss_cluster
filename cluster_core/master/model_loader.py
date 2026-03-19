"""
Загрузка модели с HuggingFace: проверка кэша, при необходимости скачивание,
разбиение state_dict на чанки для воркеров (холодная загрузка).
Поддержка побайтового прогресса через cluster_core.common.hf_download.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, List, Tuple

import torch

logger = logging.getLogger("master.model_loader")


def _state_dict_from_hf(
    model_id: str,
    progress_callback: Callable[[float, int, int, str], None] | None = None,
) -> dict:
    """Загружает state_dict модели из кэша HF или скачивает. progress_callback(percent, bytes_done, bytes_total, current_file)."""
    from cluster_core.common.hf_download import load_state_dict_from_dir
    if progress_callback is not None:
        from cluster_core.common.hf_download import download_repo_with_progress
        logger.info("Начало загрузки модели с HuggingFace (с прогрессом): %s", model_id)
        path = download_repo_with_progress(model_id, progress_callback=progress_callback)
        logger.info("Модель скачана в: %s", path)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError("Установите huggingface_hub: pip install huggingface_hub")
        logger.info("Начало загрузки модели с HuggingFace: %s", model_id)
        cache_dir = snapshot_download(repo_id=model_id, allow_patterns=["*.bin", "*.safetensors", "*.msgpack"])
        path = Path(cache_dir)
        logger.info("Модель загружена в локальное хранилище HF: %s", path)

    state_dict = load_state_dict_from_dir(path, model_id)
    logger.info("state_dict загружен из кэша: ключей=%d", len(state_dict))
    return state_dict


def _split_state_dict(state_dict: dict, n_shards: int) -> List[dict]:
    """Делит state_dict на n_shards частей по ключам (round-robin)."""
    keys = list(state_dict.keys())
    if n_shards <= 0:
        n_shards = 1
    shards: List[dict] = [{} for _ in range(n_shards)]
    for i, k in enumerate(keys):
        shards[i % n_shards][k] = state_dict[k]
    return shards


def _layer_index_from_key(key: str) -> int | None:
    """Для ключа вида 'encoder.layer.0.xxx' или 'bert.encoder.layer.0.xxx' возвращает 0."""
    for prefix in ("encoder.layer.", "bert.encoder.layer."):
        if prefix in key:
            rest = key.split(prefix, 1)[1]
            num = rest.split(".", 1)[0]
            if num.isdigit():
                return int(num)
    return None


def _layer_index_from_key_gpt2(key: str) -> int | None:
    """Для ключа вида 'transformer.h.0.xxx' возвращает 0."""
    prefix = "transformer.h."
    if prefix in key:
        rest = key.split(prefix, 1)[1]
        num = rest.split(".", 1)[0]
        if num.isdigit():
            return int(num)
    return None


def _layer_index_from_key_llama(key: str) -> int | None:
    """Для ключа вида 'model.layers.0.xxx' возвращает 0."""
    prefix = "model.layers."
    if prefix in key:
        rest = key.split(prefix, 1)[1]
        num = rest.split(".", 1)[0]
        if num.isdigit():
            return int(num)
    return None


def _layer_index_from_key_language_model_layers(key: str) -> int | None:
    """Qwen3.5 MoE / VL: 'model.language_model.layers.0.xxx' -> 0."""
    prefix = "model.language_model.layers."
    if prefix in key:
        rest = key.split(prefix, 1)[1]
        num = rest.split(".", 1)[0]
        if num.isdigit():
            return int(num)
    return None


def _split_state_dict_by_layers(
    state_dict: dict,
    n_workers: int,
    layer_index_fn: Callable[[str], int | None],
) -> List[dict]:
    """
    Группирует ключи по слоям (layer_index_fn извлекает индекс из ключа),
    распределяет слои по n_workers шардам. Ключи без слоя игнорируются.
    """
    from collections import defaultdict
    by_layer: dict[int, dict] = defaultdict(dict)
    for k, v in state_dict.items():
        idx = layer_index_fn(k)
        if idx is not None:
            by_layer[idx][k] = v
    if not by_layer:
        return []
    indices = sorted(by_layer.keys())
    n_shards = min(n_workers, len(indices))
    shards: List[dict] = [{} for _ in range(n_shards)]
    # Балансируем шардирование не по количеству ключей, а по приблизительному размеру тензоров.
    # Для больших MoE/GPTQ-моделей равное число ключей может давать сильно разный объём памяти.
    shard_sizes: List[int] = [0 for _ in range(n_shards)]
    layer_sizes: dict[int, int] = {}
    for layer_idx in indices:
        size_b = 0
        for v in by_layer[layer_idx].values():
            try:
                size_b += int(v.numel()) * int(v.element_size())
            except Exception:
                size_b += 1
        layer_sizes[layer_idx] = max(1, size_b)

    # First-fit decreasing: сначала размещаем самые "тяжёлые" слои.
    for layer_idx in sorted(indices, key=lambda x: layer_sizes.get(x, 1), reverse=True):
        shard_idx = min(range(n_shards), key=lambda i: shard_sizes[i])
        shards[shard_idx].update(by_layer[layer_idx])
        shard_sizes[shard_idx] += layer_sizes.get(layer_idx, 1)
    logger.info(
        "Layer-aware shard balancing: workers=%d layers=%d approx_sizes_mb=%s",
        n_shards,
        len(indices),
        [round(s / (1024 * 1024), 1) for s in shard_sizes],
    )
    return shards


def _split_state_dict_by_bert_layers(state_dict: dict, n_workers: int) -> List[dict]:
    """Разбиение по encoder.layer.i (BERT)."""
    return _split_state_dict_by_layers(state_dict, n_workers, _layer_index_from_key)


def _split_state_dict_by_gpt2_layers(state_dict: dict, n_workers: int) -> List[dict]:
    """Разбиение по transformer.h.i (GPT-2)."""
    return _split_state_dict_by_layers(state_dict, n_workers, _layer_index_from_key_gpt2)


def _split_state_dict_by_llama_layers(state_dict: dict, n_workers: int) -> List[dict]:
    """Разбиение по model.layers.i (LLaMA)."""
    return _split_state_dict_by_layers(state_dict, n_workers, _layer_index_from_key_llama)


def _split_state_dict_by_language_model_layers(state_dict: dict, n_workers: int) -> List[dict]:
    """Разбиение по model.language_model.layers.i (Qwen3.5 MoE и др.)."""
    return _split_state_dict_by_layers(state_dict, n_workers, _layer_index_from_key_language_model_layers)


def prepare_shards(hf_model_id: str, n_workers: int) -> List[bytes]:
    """
    Обеспечивает наличие модели в кэше HF, загружает state_dict,
    разбивает на n_workers чанков, сериализует каждый в bytes (torch.save).
    Возвращает список из n_workers байтовых блобов.
    """
    if n_workers <= 0:
        raise ValueError("n_workers должно быть >= 1")
    state_dict = _state_dict_from_hf(hf_model_id)
    keys = list(state_dict.keys())
    logger.info("Начало разбивки state_dict по слоям для n_workers=%d", n_workers)
    # Выбор разбиения по слоям по префиксам ключей
    if any("transformer.h." in k for k in keys):
        shard_dicts = _split_state_dict_by_gpt2_layers(state_dict, n_workers)
        shard_type = "GPT-2 (transformer.h)"
    elif any("model.language_model.layers." in k for k in keys):
        shard_dicts = _split_state_dict_by_language_model_layers(state_dict, n_workers)
        shard_type = "language_model.layers (Qwen3.5 MoE / VL)"
    elif any("model.layers." in k for k in keys):
        shard_dicts = _split_state_dict_by_llama_layers(state_dict, n_workers)
        shard_type = "LLaMA (model.layers)"
    elif any("encoder.layer." in k or "bert.encoder.layer." in k for k in keys):
        shard_dicts = _split_state_dict_by_bert_layers(state_dict, n_workers)
        shard_type = "BERT (encoder.layer)"
    else:
        shard_dicts = []
        shard_type = None
    if not shard_dicts:
        shard_dicts = _split_state_dict(state_dict, n_workers)
    else:
        logger.info("Sharding by %s for %s", shard_type or "layers", hf_model_id)
        while len(shard_dicts) < n_workers:
            shard_dicts.append({})
        shard_dicts = shard_dicts[:n_workers]
    result: List[bytes] = []
    buf = io.BytesIO()
    for sd in shard_dicts:
        buf.seek(0)
        buf.truncate()
        torch.save(sd, buf)
        result.append(buf.getvalue())
    sizes = [len(r) for r in result]
    logger.info(
        "Чанки подготовлены и сериализованы: %d штук для модели %s, размеры (байт): %s, суммарно %.1f МБ",
        len(result), hf_model_id, sizes, sum(sizes) / (1024 * 1024),
    )
    return result


def prepare_shard_keys(
    hf_model_id: str,
    n_workers: int,
    progress_callback: Callable[[float, int, int, str], None] | None = None,
) -> List[List[str]]:
    """
    Подготавливает план шардирования (только ключи).
    Мастер и воркеры будут скачивать веса из HF, а воркер загрузит только свой поднабор ключей.
    progress_callback(percent, bytes_done, bytes_total, current_file) для побайтового прогресса на мастере.
    """
    if n_workers <= 0:
        raise ValueError("n_workers должно быть >= 1")
    state_dict = _state_dict_from_hf(hf_model_id, progress_callback=progress_callback)
    keys = list(state_dict.keys())
    logger.info("Подготовка меток (ключей) для шардов: keys=%d, workers=%d", len(keys), n_workers)

    if any("transformer.h." in k for k in keys):
        shard_dicts = _split_state_dict_by_gpt2_layers(state_dict, n_workers)
        shard_type = "GPT-2 (transformer.h)"
    elif any("model.language_model.layers." in k for k in keys):
        shard_dicts = _split_state_dict_by_language_model_layers(state_dict, n_workers)
        shard_type = "language_model.layers (Qwen3.5 MoE / VL)"
    elif any("model.layers." in k for k in keys):
        shard_dicts = _split_state_dict_by_llama_layers(state_dict, n_workers)
        shard_type = "LLaMA (model.layers)"
    elif any("encoder.layer." in k or "bert.encoder.layer." in k for k in keys):
        shard_dicts = _split_state_dict_by_bert_layers(state_dict, n_workers)
        shard_type = "BERT (encoder.layer)"
    else:
        shard_dicts = []
        shard_type = None

    if not shard_dicts:
        shard_dicts = _split_state_dict(state_dict, n_workers)
    else:
        logger.info("Sharding by %s for %s (keys-only)", shard_type or "layers", hf_model_id)
        while len(shard_dicts) < n_workers:
            shard_dicts.append({})
        shard_dicts = shard_dicts[:n_workers]

    # Освобождаем память: нам нужны только ключи
    key_lists: List[List[str]] = [sorted(sd.keys()) for sd in shard_dicts]
    logger.info("Подготовлены shard_keys для %d шардов (пример: shard0=%d keys)", len(key_lists), len(key_lists[0]) if key_lists else 0)
    return key_lists


def prepare_layer_ranges(
    hf_model_id: str,
    n_workers: int,
    progress_callback: Callable[[float, int, int, str], None] | None = None,
) -> List[Tuple[int, int]]:
    """
    План для режима streaming_chunks: вычисляет диапазоны слоёв для воркеров без загрузки всего state_dict в RAM.
    1) Скачивает safetensors в кэш (с прогрессом, если задан callback)
    2) Сканирует только ключи в .safetensors и определяет индексы слоёв (model.layers.N.)
    3) Делит слои на n_workers непрерывными диапазонами [start, end) (end не включительно)
    """
    if n_workers <= 0:
        raise ValueError("n_workers должно быть >= 1")
    from cluster_core.common.hf_download import download_repo_with_progress

    path = download_repo_with_progress(hf_model_id, progress_callback=progress_callback) if progress_callback else download_repo_with_progress(hf_model_id)

    st_files = sorted(Path(path).glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"Для streaming_chunks нужны .safetensors, но в {path} их нет")
    try:
        import re as _re
        from safetensors import safe_open
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Для streaming_chunks нужен пакет safetensors: {e}") from e

    re_lm = _re.compile(r"^model\.language_model\.layers\.(\d+)\.")
    re_plain = _re.compile(r"^model\.layers\.(\d+)\.")
    indices: set[int] = set()
    for sf in st_files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for k in f.keys():
                m = re_lm.match(k) or re_plain.match(k)
                if m:
                    indices.add(int(m.group(1)))
    if not indices:
        raise RuntimeError(
            "Не удалось определить слои по ключам (ожидались model.layers.N. или model.language_model.layers.N.)."
        )
    ordered = sorted(indices)
    layer_min, layer_max = ordered[0], ordered[-1]
    total_layers = layer_max - layer_min + 1
    # Равномерно делим непрерывный диапазон на n_workers
    n = min(n_workers, total_layers)
    base = total_layers // n
    rem = total_layers % n
    ranges: List[Tuple[int, int]] = []
    cur = layer_min
    for i in range(n):
        size = base + (1 if i < rem else 0)
        ranges.append((cur, cur + size))
        cur += size
    # Если воркеров больше слоёв, последние диапазоны будут пустыми; не выдаём их.
    logger.info("Streaming plan: layers=%d (min=%d max=%d) workers=%d ranges=%s", total_layers, layer_min, layer_max, n_workers, ranges)
    return ranges
