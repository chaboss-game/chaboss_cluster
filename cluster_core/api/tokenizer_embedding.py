"""
Токенизация и получение эмбеддингов для HTTP API.
Кэш по model_id: tokenizer и модуль эмбеддингов (для BERT — только embedding layer).
Вход pipeline — скрытые состояния [batch, seq_len, hidden_size] после эмбеддингов.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

import torch

logger = logging.getLogger("master.tokenizer_embedding")

# Кэш: model_id -> (tokenizer, embedding_module)
_tokenizer_cache: dict[str, Any] = {}
_embedding_cache: dict[str, torch.nn.Module] = {}
# Ограничение длины последовательности по умолчанию
DEFAULT_MAX_LENGTH = 512


def _get_bert_embedding_module(model_id: str) -> torch.nn.Module | None:
    """
    Загружает BERT-модель, оставляет только слой эмбеддингов, остальное удаляет.
    Возвращает модуль для преобразования input_ids -> hidden_states [B, seq, hidden_size].
    """
    try:
        from transformers import BertModel
    except ImportError:
        logger.warning("transformers не установлен")
        return None
    try:
        model = BertModel.from_pretrained(model_id)
        emb = model.bert.embeddings
        # Удаляем тяжёлые части, оставляем только эмбеддинги
        del model.bert.encoder
        del model.pooler
        del model.bert
        del model
        emb.eval()
        return emb
    except Exception as e:
        logger.warning("Не удалось загрузить BERT embeddings для %s: %s", model_id, e)
        return None


def get_tokenizer(model_id: str):
    """Возвращает кэшированный токенизатор для model_id."""
    if model_id not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer
            _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logger.warning("Не удалось загрузить токенизатор для %s: %s", model_id, e)
            return None
    return _tokenizer_cache.get(model_id)


def get_embedding_module(model_id: str) -> torch.nn.Module | None:
    """Возвращает кэшированный модуль эмбеддингов для model_id (BERT)."""
    if model_id not in _embedding_cache:
        mod = _get_bert_embedding_module(model_id)
        if mod is not None:
            _embedding_cache[model_id] = mod
    return _embedding_cache.get(model_id)


def text_to_embeddings(
    model_id: str,
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "cpu",
) -> Tuple[torch.Tensor | None, int]:
    """
    Токенизирует текст и возвращает тензор эмбеддингов [1, seq_len, hidden_size]
    и число токенов (prompt_tokens).
    При ошибке возвращает (None, 0).
    """
    tokenizer = get_tokenizer(model_id)
    emb_module = get_embedding_module(model_id)
    if tokenizer is None or emb_module is None:
        return None, 0

    try:
        enc = tokenizer(
            text or " ",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        attention_mask = enc.get("attention_mask")

        with torch.no_grad():
            emb_module.eval()
            hidden = emb_module(
                input_ids=input_ids.to(device),
                token_type_ids=token_type_ids.to(device),
            )
        # hidden: [batch, seq, hidden_size]
        n_tokens = input_ids.shape[1]
        return hidden.cpu(), n_tokens
    except Exception as e:
        logger.exception("text_to_embeddings failed: %s", e)
        return None, 0


def clear_cache(model_id: str | None = None) -> None:
    """Очищает кэш токенизатора и эмбеддингов (для model_id или полностью)."""
    if model_id is None:
        _tokenizer_cache.clear()
        _embedding_cache.clear()
    else:
        _tokenizer_cache.pop(model_id, None)
        _embedding_cache.pop(model_id, None)
