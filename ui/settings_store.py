"""
Хранение настроек GUI: автосохранение в JSON.
Файл: ~/.config/chaboss_cluster/gui_settings.json или CHABOSS_CLUSTER_SETTINGS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List


def _config_path() -> Path:
    path = os.environ.get("CHABOSS_CLUSTER_SETTINGS")
    if path:
        return Path(path)
    return Path.home() / ".config" / "chaboss_cluster" / "gui_settings.json"


DEFAULT = {
    "master_addr": "127.0.0.1:60051",
    "hf_model_id": "",
    "workers": [
        {"host": "127.0.0.1", "port": 60052, "auth_token": ""},
    ],
}


def load() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return dict(DEFAULT)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return dict(DEFAULT)
    # merge with defaults so new keys appear
    out = dict(DEFAULT)
    out.update(data)
    if "workers" in data and isinstance(data["workers"], list):
        out["workers"] = [
            {"host": str(w.get("host", "")), "port": int(w.get("port", 0)), "auth_token": str(w.get("auth_token", ""))}
            for w in data["workers"]
        ]
    return out


def save(data: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
