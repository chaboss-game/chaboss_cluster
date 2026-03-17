"""
Скачивание репозитория HuggingFace с побайтовым прогрессом (httpx + chunked).
Используется для отображения «живого» процента загрузки в GUI.
На воркере всегда используем pathlib.Path через модуль pathlib, чтобы избежать NameError.
"""
from __future__ import annotations

import logging
import pathlib
import re
from typing import Callable

logger = logging.getLogger("cluster.hf_download")

# Паттерны файлов весов (как в model_loader)
ALLOW_PATTERNS = ("*.bin", "*.safetensors", "*.msgpack")
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _match_patterns(filename: str) -> bool:
    for pat in ALLOW_PATTERNS:
        if pat.startswith("*."):
            if filename.endswith(pat[1:]):
                return True
    return False


def download_repo_with_progress(
    repo_id: str,
    revision: str = "main",
    cache_dir: str | pathlib.Path | None = None,
    token: str | bool | None = None,
    progress_callback: Callable[[float, int, int, str], None] | None = None,
) -> pathlib.Path:
    """
    Скачивает файлы репозитория (*.bin, *.safetensors, *.msgpack) в локальную папку
    с вызовом progress_callback(percent, bytes_done, bytes_total, current_file).

    Возвращает путь к скачанной директории (в ней лежат файлы в корне).
    """
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    except ImportError:
        raise RuntimeError("Установите huggingface_hub: pip install huggingface_hub")

    try:
        import httpx
    except ImportError:
        raise RuntimeError("Установите httpx: pip install httpx")

    api = HfApi()
    all_files = api.list_repo_files(repo_id, revision=revision)
    files = [f for f in all_files if _match_patterns(f)]
    if not files:
        raise FileNotFoundError(f"В репозитории {repo_id} не найдено файлов {ALLOW_PATTERNS}")

    # Метаданные и общий размер
    file_infos: list[tuple[str, str, int]] = []  # (filename, download_url, size)
    total_size = 0
    for filename in files:
        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        meta = get_hf_file_metadata(url, token=token)
        if meta.size is None:
            meta.size = 0
        file_infos.append((filename, meta.location or url, meta.size))
        total_size += meta.size

    if total_size == 0:
        total_size = 1  # избежать деления на 0

    # Директория кэша: cache_dir/chaboss/<repo_sanitized>/<revision>/
    cache_base = pathlib.Path(cache_dir) if cache_dir else pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    repo_safe = re.sub(r"[^\w\-.]", "--", repo_id)
    out_dir = cache_base / "chaboss_downloads" / repo_safe / revision
    out_dir.mkdir(parents=True, exist_ok=True)

    bytes_done = 0
    headers = {}
    if token and isinstance(token, str):
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(follow_redirects=True, timeout=60.0, headers=headers or None) as client:
        for filename, download_url, size in file_infos:
            out_path = out_dir / pathlib.Path(filename).name
            if out_path.exists() and out_path.stat().st_size == size:
                bytes_done += size
                if progress_callback:
                    progress_callback(100.0 * bytes_done / total_size, bytes_done, total_size, filename)
                continue

            logger.info("Скачивание %s (%s)", filename, _fmt_size(size))
            with client.stream("GET", download_url) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        bytes_done += len(chunk)
                        if progress_callback:
                            progress_callback(100.0 * bytes_done / total_size, bytes_done, total_size, filename)

    return out_dir


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def load_state_dict_from_dir(path: pathlib.Path | str, model_id: str = ""):
    """
    Загружает state_dict из директории (pytorch_model.bin / model.safetensors / *.safetensors).
    Возвращает dict. model_id используется только для fallback через transformers.
    """
    import torch
    path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
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
            raise RuntimeError("Установите safetensors: pip install safetensors")
    if state_dict is None:
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
    if state_dict is None and model_id:
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_id)
            state_dict = model.state_dict()
            del model
        except Exception as e:
            raise FileNotFoundError(f"Не найдены веса в {path} и не удалось загрузить через transformers: {e}") from e
    if state_dict is None:
        raise FileNotFoundError(f"Не найдены веса в {path}")
    return state_dict
