"""
Загрузка модели с HuggingFace: проверка кэша, при необходимости скачивание,
разбиение state_dict на чанки для воркеров (холодная загрузка).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, List

import torch

logger = logging.getLogger("master.model_loader")


def _state_dict_from_hf(model_id: str) -> dict:
    """Загружает state_dict модели из кэша HF или скачивает. Возвращает state_dict."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("Установите huggingface_hub: pip install huggingface_hub")

    logger.info("Начало загрузки модели с HuggingFace: %s", model_id)
    cache_dir = snapshot_download(repo_id=model_id, allow_patterns=["*.bin", "*.safetensors", "*.msgpack"])
    path = Path(cache_dir)
    logger.info("Модель загружена в локальное хранилище HF: %s", path)

    # Пробуем pytorch_model.bin или model.safetensors
    state_dict = None
    if (path / "pytorch_model.bin").exists():
        state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            state_dict = getattr(state_dict, "state_dict", lambda: state_dict)()
    elif (path / "model.safetensors").exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(path / "model.safetensors"))
        except ImportError:
            raise RuntimeError("Для .safetensors установите safetensors: pip install safetensors")

    if state_dict is None:
        # Один большой .safetensors или несколько файлов
        st_files = list(path.glob("*.safetensors"))
        if st_files:
            try:
                from safetensors.torch import load_file
                state_dict = {}
                for f in st_files:
                    state_dict.update(load_file(str(f)))
            except ImportError:
                raise RuntimeError("Установите safetensors: pip install safetensors")
        else:
            bin_files = list(path.glob("*.bin"))
            if bin_files:
                state_dict = torch.load(bin_files[0], map_location="cpu", weights_only=True)
                if not isinstance(state_dict, dict):
                    state_dict = getattr(state_dict, "state_dict", lambda: state_dict)()

    if state_dict is None:
        # Fallback: загрузить через transformers и взять state_dict (тяжело для больших моделей)
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_id)
            state_dict = model.state_dict()
            del model
        except Exception as e:
            raise FileNotFoundError(
                f"В репозитории {model_id} не найдены веса и не удалось загрузить через transformers: {e}"
            ) from e
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
    for i, layer_idx in enumerate(indices):
        shard_idx = i % n_shards
        shards[shard_idx].update(by_layer[layer_idx])
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
    # Выбор разбиения по слоям по префиксам ключей (приоритет: GPT-2, LLaMA, BERT)
    if any("transformer.h." in k for k in keys):
        shard_dicts = _split_state_dict_by_gpt2_layers(state_dict, n_workers)
        shard_type = "GPT-2 (transformer.h)"
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
