"""
Потокобезопасный кольцевой буфер логов для отображения в GUI.
Используется мастером и воркерами: логи пишутся в файл и в буфер.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Tuple


class LogBuffer:
    """Буфер последних max_lines строк. add() и get_since() потокобезопасны."""

    def __init__(self, max_lines: int = 2000) -> None:
        self._max_lines = max(100, max_lines)
        self._lines: List[str] = []
        self._lock = threading.Lock()
        self._next_index = 0

    def add(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self._max_lines:
                self._lines.pop(0)
            self._next_index = len(self._lines)

    def get_since(self, since_index: int) -> Tuple[List[str], int]:
        """Возвращает (list of lines, next_index). since_index 0 = все доступные."""
        with self._lock:
            n = len(self._lines)
            if since_index >= n:
                return [], n
            result = self._lines[since_index:n]
            return result, n


def make_buffer_handler(buffer: LogBuffer, fmt: str | None = None) -> logging.Handler:
    """Возвращает logging.Handler, который пишет в buffer.add()."""
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    class BufferHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                buffer.add(formatter.format(record))
            except Exception:
                pass

    return BufferHandler()
