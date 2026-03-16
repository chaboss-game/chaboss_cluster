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
    return MasterConfig(
        listen_host=data.get("listen_host", "0.0.0.0"),
        listen_port=int(data.get("listen_port", 50051)),
        workers=workers,
        parallel_backend=data.get("parallel_backend", "baseline"),
    )


def load_worker_config(path: str | Path) -> dict:
    return _load_yaml(path)


def _load_yaml(path: str | Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

