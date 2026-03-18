from __future__ import annotations

"""
Локальное хранение чата/вложений в папке ./cluster_shared/chat/.

Форматы:
- channels.json: список/мапа каналов
- messages.jsonl: append-only история (JSONL), каждая строка — сообщение
- attachments/<attachment_id>/original + (опционально) thumbnails/<attachment_id>.jpg
- received_messages/<message_id>.done (для идемпотентности на воркерах)
"""

import base64
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


@dataclass(frozen=True)
class StoredChatAttachment:
    attachment_id: str
    filename: str
    mime_type: str
    is_image: bool
    size: int
    # храним thumbnail как base64-строку (маленькое), чтобы message jsonl оставался self-contained
    thumbnail_jpeg_b64: str | None = None


class ChatStorage:
    def __init__(
        self,
        shared_root: Path,
        *,
        cache_messages: int = 2000,
    ) -> None:
        self._lock = threading.RLock()
        self._cache_messages = max(200, cache_messages)

        self._shared_root = shared_root
        self._chat_root = self._shared_root / "chat"
        self._channels_path = self._chat_root / "channels.json"
        self._messages_path = self._chat_root / "messages.jsonl"
        self._attachments_dir = self._chat_root / "attachments"
        self._thumbnails_dir = self._chat_root / "thumbnails"
        self._received_dir = self._chat_root / "received_messages"

        self._channels: Dict[str, str] = {}  # id -> name
        self._seq_counter = 0
        self._cache: Deque[Dict[str, Any]] = deque(maxlen=self._cache_messages)
        self._min_seq_in_cache = 0

        self._init_files()
        self._load()

    def _init_files(self) -> None:
        self._chat_root.mkdir(parents=True, exist_ok=True)
        self._attachments_dir.mkdir(parents=True, exist_ok=True)
        self._thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self._received_dir.mkdir(parents=True, exist_ok=True)

        if not self._channels_path.exists():
            _safe_json_dump(self._channels_path, [{"id": "general", "name": "general"}])

        if not self._messages_path.exists():
            self._messages_path.write_text("", encoding="utf-8")

    def _load(self) -> None:
        with self._lock:
            # channels
            try:
                channels_raw = json.loads(self._channels_path.read_text(encoding="utf-8") or "[]")
            except Exception:
                channels_raw = [{"id": "general", "name": "general"}]
            self._channels = {}
            for item in channels_raw:
                cid = str(item.get("id") or "").strip()
                name = str(item.get("name") or "").strip()
                if cid and name:
                    self._channels[cid] = name
            if not self._channels:
                self._channels = {"general": "general"}
                _safe_json_dump(self._channels_path, [{"id": "general", "name": "general"}])

            # messages + seq counter + cache tail
            self._seq_counter = 0
            self._cache.clear()

            # Читаем хвост файла — держим только последние N сообщений в памяти.
            if self._messages_path.stat().st_size == 0:
                self._min_seq_in_cache = 0
                return

            # Ограниченная загрузка в память: читаем всё только в крайнем случае.
            # Для JSONL хвоста обычно достаточно прохода потоком с deque.
            last_seq = 0
            with self._messages_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except Exception:
                        continue
                    seq = int(msg.get("seq") or 0)
                    if seq > 0:
                        last_seq = seq
                    self._cache.append(msg)

            self._seq_counter = last_seq
            if self._cache:
                self._min_seq_in_cache = int(self._cache[0].get("seq") or 0)
            else:
                self._min_seq_in_cache = 0

    def list_channels(self) -> List[Tuple[str, str]]:
        with self._lock:
            return [(cid, name) for cid, name in self._channels.items()]

    def mutate_channels(self, ops: Iterable[Dict[str, Any]]) -> None:
        with self._lock:
            for op in ops:
                t = op.get("type")
                channel_id = str(op.get("channel_id") or "").strip()
                name = str(op.get("name") or "").strip()

                if t == "create":
                    if not channel_id:
                        raise ValueError("channel_id required for create")
                    if not name:
                        raise ValueError("name required for create")
                    # Идемпотентность: если канал уже существует — обновим имя, но не падём.
                    self._channels[channel_id] = name
                elif t == "rename":
                    # Идемпотентность: если канала нет — ничего не делаем.
                    if not channel_id or channel_id not in self._channels:
                        continue
                    if not name:
                        continue
                    self._channels[channel_id] = name
                elif t == "delete":
                    # Идемпотентность: если канала нет — ничего не делаем.
                    if not channel_id or channel_id not in self._channels:
                        continue
                    del self._channels[channel_id]
                else:
                    raise ValueError(f"unknown mutation type: {t}")

            # Бэкап: канал general должен существовать всегда.
            if "general" not in self._channels:
                self._channels["general"] = "general"

            channels_arr = [{"id": cid, "name": name} for cid, name in self._channels.items()]
            _safe_json_dump(self._channels_path, channels_arr)

    def _attachment_dir(self, attachment_id: str) -> Path:
        return self._attachments_dir / attachment_id

    def attachment_original_path(self, attachment_id: str) -> Path:
        return self._attachment_dir(attachment_id) / "original"

    def attachment_thumbnail_path(self, attachment_id: str) -> Path:
        return self._thumbnails_dir / f"{attachment_id}.jpg"

    def received_marker_path(self, message_id: str) -> Path:
        return self._received_dir / f"{message_id}.done"

    def has_received_message(self, message_id: str) -> bool:
        return self.received_marker_path(message_id).exists()

    def save_received_marker(self, message_id: str) -> None:
        marker = self.received_marker_path(message_id)
        marker.write_text("ok", encoding="utf-8")

    def write_attachment_bytes(
        self,
        attachment_id: str,
        original_bytes: bytes,
    ) -> None:
        p = self.attachment_original_path(attachment_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(original_bytes)
        tmp.replace(p)

    def write_attachment_stream_to_file(
        self,
        attachment_id: str,
        data_iter: Iterable[bytes],
    ) -> None:
        p = self.attachment_original_path(attachment_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for chunk in data_iter:
                if not chunk:
                    continue
                f.write(chunk)
        tmp.replace(p)

    def write_thumbnail_bytes(self, attachment_id: str, thumbnail_jpeg: bytes) -> None:
        p = self.attachment_thumbnail_path(attachment_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(thumbnail_jpeg)
        tmp.replace(p)

    def append_message(
        self,
        *,
        message_id: str,
        timestamp_ms: int,
        channel_id: str,
        sender: str,
        text: str,
        attachments: List[StoredChatAttachment],
    ) -> int:
        with self._lock:
            self._seq_counter += 1
            seq = self._seq_counter

            msg = {
                "message_id": message_id,
                "seq": seq,
                "timestamp_ms": int(timestamp_ms),
                "channel_id": channel_id,
                "sender": sender,
                "text": text or "",
                "attachments": [
                    {
                        "attachment_id": a.attachment_id,
                        "filename": a.filename,
                        "mime_type": a.mime_type,
                        "is_image": bool(a.is_image),
                        "size": int(a.size),
                        "thumbnail_jpeg_b64": a.thumbnail_jpeg_b64,
                    }
                    for a in attachments
                ],
            }

            # append jsonl
            with self._messages_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            self._cache.append(msg)
            if len(self._cache) == self._cache.maxlen:
                self._min_seq_in_cache = int(self._cache[0].get("seq") or 0)
            else:
                if self._min_seq_in_cache == 0:
                    self._min_seq_in_cache = int(msg.get("seq") or 0)

            return seq

    def get_history(
        self,
        *,
        channel_id: str,
        since_seq: int,
        limit: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            since_seq = max(0, int(since_seq))
            limit = max(1, min(2000, int(limit)))
            next_seq = self._seq_counter

            def filter_from_cache() -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for msg in reversed(self._cache):
                    if msg.get("channel_id") != channel_id:
                        continue
                    seq = int(msg.get("seq") or 0)
                    if seq <= since_seq:
                        continue
                    out.append(msg)
                    if len(out) >= limit:
                        break
                out.reverse()
                return out

            cached_msgs = filter_from_cache()
            if cached_msgs:
                return cached_msgs, next_seq

            # fallback: since_seq может быть меньше, чем min seq в cache
            if since_seq >= self._min_seq_in_cache:
                return [], next_seq

            out: List[Dict[str, Any]] = []
            with self._messages_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except Exception:
                        continue
                    if msg.get("channel_id") != channel_id:
                        continue
                    seq = int(msg.get("seq") or 0)
                    if seq <= since_seq:
                        continue
                    out.append(msg)
                    if len(out) >= limit:
                        break

            return out, next_seq

    def read_attachment_bytes(self, attachment_id: str) -> bytes:
        p = self.attachment_original_path(attachment_id)
        return p.read_bytes()

    def read_thumbnail_bytes(self, attachment_id: str) -> bytes | None:
        p = self.attachment_thumbnail_path(attachment_id)
        if not p.exists():
            return None
        return p.read_bytes()

