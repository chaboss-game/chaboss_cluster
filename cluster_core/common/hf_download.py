"""
Скачивание репозитория HuggingFace с побайтовым прогрессом (httpx + chunked).
Используется для отображения «живого» процента загрузки в GUI.
На воркере всегда используем pathlib.Path через модуль pathlib, чтобы избежать NameError.
"""
from __future__ import annotations

import logging
import pathlib
import re
from typing import Callable, Iterable

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


def load_state_dict_from_dir(
    path: pathlib.Path | str,
    model_id: str = "",
    keys: Iterable[str] | None = None,
):
    """
    Загружает state_dict из директории (pytorch_model.bin / model.safetensors / *.safetensors).
    Возвращает dict. model_id используется только для fallback через transformers.
    Если передан keys — пытается загрузить только указанные тензоры (эффективно для больших моделей).
    """
    import torch
    path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
    state_dict = None
    keys_list = list(keys) if keys is not None else None
    if keys_list is not None:
        logger.info(
            "load_state_dict_from_dir: path=%s model_id=%s keys_filter=ON requested=%d",
            str(path),
            model_id or "",
            len(keys_list),
        )
    if (path / "pytorch_model.bin").exists():
        logger.info("load_state_dict_from_dir: loading pytorch_model.bin from %s", str(path))
        state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            state_dict = getattr(state_dict, "state_dict", lambda: state_dict)()
    elif (path / "model.safetensors").exists():
        try:
            from safetensors import safe_open
            # Если keys задан, читаем только нужные тензоры без загрузки всей модели в RAM.
            if keys_list is not None:
                want = set(keys_list)
                state_dict = {}
                logger.info("load_state_dict_from_dir: reading model.safetensors filtered wanted=%d", len(want))
                with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k in want:
                            state_dict[k] = f.get_tensor(k)
                logger.info("load_state_dict_from_dir: model.safetensors filtered found=%d", len(state_dict))
                # Если в этом файле ничего не нашли — оставим пустой dict, дальше не падаем.
            else:
                from safetensors.torch import load_file
                logger.info("load_state_dict_from_dir: reading full model.safetensors (may be heavy)")
                state_dict = load_file(str(path / "model.safetensors"))
        except ImportError:
            raise RuntimeError("Установите safetensors: pip install safetensors")
    if state_dict is None:
        st_files = list(path.glob("*.safetensors"))
        if st_files:
            try:
                from safetensors import safe_open
                state_dict = {}
                want = set(keys_list) if keys_list is not None else None
                for sf in st_files:
                    if want is None:
                        from safetensors.torch import load_file
                        logger.info("load_state_dict_from_dir: reading full safetensors file=%s", str(sf))
                        state_dict.update(load_file(str(sf)))
                        continue
                    logger.info(
                        "load_state_dict_from_dir: reading safetensors filtered file=%s wanted=%d",
                        str(sf),
                        len(want),
                    )
                    before = len(state_dict)
                    with safe_open(str(sf), framework="pt", device="cpu") as f:
                        # Итерируем ключи файла (без загрузки тензоров) и забираем только нужные.
                        for k in f.keys():
                            if k in want:
                                state_dict[k] = f.get_tensor(k)
                    logger.info(
                        "load_state_dict_from_dir: safetensors filtered file=%s found_new=%d total_now=%d",
                        str(sf),
                        len(state_dict) - before,
                        len(state_dict),
                    )
            except ImportError:
                raise RuntimeError("Установите safetensors: pip install safetensors")
        else:
            bin_files = list(path.glob("*.bin"))
            if bin_files:
                logger.info("load_state_dict_from_dir: loading .bin files from %s (first=%s)", str(path), str(bin_files[0]))
                state_dict = torch.load(bin_files[0], map_location="cpu", weights_only=True)
                if not isinstance(state_dict, dict):
                    state_dict = getattr(state_dict, "state_dict", lambda: state_dict)()
    if state_dict is not None and keys_list is not None:
        # Для .bin (и некоторых нетипичных случаев) всё равно могло загрузиться много — фильтруем.
        want = set(keys_list)
        if not isinstance(state_dict, dict):
            raise TypeError("Ожидался state_dict (dict)")
        state_dict = {k: v for k, v in state_dict.items() if k in want}

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
