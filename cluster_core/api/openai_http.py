"""
OpenAI-совместимый HTTP API для интеграции с OpenCLaw и другими клиентами.
Эндпоинты: /v1/models, /v1/chat/completions, /v1/completions.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from cluster_core.api.tokenizer_embedding import text_to_embeddings

logger = logging.getLogger("master.openai_http")


# --- Request/Response models (OpenAI-like) ---

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="", description="ID модели")
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = Field(default="")
    prompt: str | List[str] | List[int] = ""
    max_tokens: Optional[int] = None
    stream: bool = False


def create_app(registry: Any, master_node: Any, config: Any = None) -> FastAPI:
    """Создаёт FastAPI-приложение с доступом к реестру, мастеру и конфигу (openai_api_key)."""
    app = FastAPI(title="Chaboss Cluster OpenAI API", version="0.1.0")
    app.state.registry = registry
    app.state.master_node = master_node
    app.state.config = config

    def _check_api_key(authorization: str | None, api_key: str | None) -> None:
        if not api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization[7:].strip()
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _get_model_id(request: Request) -> str:
        master_node = getattr(request.app.state, "master_node", None)
        if master_node and hasattr(master_node, "get_last_loaded_model_id"):
            model_id = master_node.get_last_loaded_model_id()
            if model_id:
                return model_id
        return "cluster-default"

    @app.get("/v1/models")
    async def list_models(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict:
        """Список доступных моделей (последняя загруженная через LoadModel)."""
        cfg = getattr(request.app.state, "config", None)
        api_key = getattr(cfg, "openai_api_key", None) if cfg else None
        _check_api_key(authorization, api_key)

        model_id = _get_model_id(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "chaboss-cluster",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict:
        """Чат-завершение: при загруженной модели прогоняет тензор по pipeline воркеров."""
        cfg = getattr(request.app.state, "config", None)
        api_key = getattr(cfg, "openai_api_key", None) if cfg else None
        _check_api_key(authorization, api_key)

        model_id = body.model or _get_model_id(request)
        content = ""
        if body.messages:
            last = body.messages[-1]
            content = last.content if hasattr(last, "content") else str(last)

        prompt_tokens = 0
        completion_tokens = 0
        master_node = getattr(request.app.state, "master_node", None)
        if master_node and hasattr(master_node, "run_pipeline") and master_node.get_last_loaded_model_id():
            import asyncio

            loaded_id = master_node.get_last_loaded_model_id()
            # Токенизация и эмбеддинги через кэшированный токенизатор/эмбеддинг-слой
            embeddings_tensor, prompt_tokens = text_to_embeddings(
                loaded_id, content or " ", device="cpu"
            )
            if embeddings_tensor is not None:
                try:
                    loop = asyncio.get_running_loop()
                    out_tensor = await loop.run_in_executor(
                        None,
                        lambda: master_node.run_pipeline(embeddings_tensor),
                    )
                    # BERT не генерирует текст; возвращаем факт прохода по pipeline и форму выхода
                    content = (
                        f"[Энкодер: обработано токенов {prompt_tokens}, выход shape: {tuple(out_tensor.shape)}] "
                        + (content or "")
                    )
                except Exception as e:
                    logger.exception("run_pipeline failed")
                    content = f"[Pipeline error: {e}] " + (content or "")
            else:
                content = "[Токенизатор/эмбеддинги недоступны для данной модели] " + (content or "")

        if not content:
            content = "(нет сообщений или модель не загружена)"

        total_tokens = prompt_tokens + completion_tokens
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    @app.post("/v1/completions")
    async def completions(
        body: CompletionRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict:
        """Text completion (заглушка)."""
        cfg = getattr(request.app.state, "config", None)
        api_key = getattr(cfg, "openai_api_key", None) if cfg else None
        _check_api_key(authorization, api_key)

        model_id = body.model or _get_model_id(request)
        prompt = body.prompt if isinstance(body.prompt, str) else ""
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "text": prompt or "(заглушка)",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
