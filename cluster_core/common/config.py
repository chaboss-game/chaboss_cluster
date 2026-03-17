from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class WorkerConfig:
    host: str
    port: int
    auth_token: str | None = None


@dataclass
class MasterConfig:
    listen_host: str
    listen_port: int
    workers: List[WorkerConfig]
    parallel_backend: str = "baseline"  # baseline | accelerate | deepspeed
    # Режим загрузки модели: fit_in_cluster | streaming_chunks
    model_load_mode: str = "fit_in_cluster"
    resource_usage_percent: int = 75  # использование свободных VRAM/RAM под модель, 1–100
    # OpenAI-совместимый HTTP API
    http_listen_host: str = "0.0.0.0"
    http_listen_port: int = 8055
    openai_api_key: str | None = None  # опционально; если задан — проверка заголовка Authorization


def load_master_config(path: str | Path) -> MasterConfig:
    data = _load_yaml(path)
    workers = [
        WorkerConfig(
            host=w["host"],
            port=int(w["port"]),
            auth_token=w.get("auth_token"),
        )
        for w in data.get("workers", [])
    ]
    rup = data.get("resource_usage_percent", 75)
    rup = max(1, min(100, int(rup)))
    return MasterConfig(
        listen_host=data.get("listen_host", "0.0.0.0"),
        listen_port=int(data.get("listen_port", 50051)),
        workers=workers,
        parallel_backend=data.get("parallel_backend", "baseline"),
        model_load_mode=data.get("model_load_mode", "fit_in_cluster"),
        resource_usage_percent=rup,
        http_listen_host=data.get("http_listen_host", "0.0.0.0"),
        http_listen_port=int(data.get("http_listen_port", 8055)),
        openai_api_key=data.get("openai_api_key") or None,
    )


def load_worker_config(path: str | Path) -> dict:
    return _load_yaml(path)


def _load_yaml(path: str | Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

