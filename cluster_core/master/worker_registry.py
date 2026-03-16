from __future__ import annotations

import threading
from typing import Dict, Optional

from cluster_core.common.types import WorkerDescriptor, WorkerId, WorkerStatus


class WorkerRegistry:
    """
    Thread-safe in-memory registry of all known workers and their state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: Dict[str, WorkerDescriptor] = {}

    def upsert(self, descriptor: WorkerDescriptor) -> None:
        with self._lock:
            self._workers[descriptor.worker_id.as_str()] = descriptor

    def get(self, worker_id: WorkerId) -> Optional[WorkerDescriptor]:
        with self._lock:
            return self._workers.get(worker_id.as_str())

    def all(self) -> Dict[str, WorkerDescriptor]:
        with self._lock:
            return dict(self._workers)

    def set_status(self, worker_id: WorkerId, status: WorkerStatus) -> None:
        with self._lock:
            key = worker_id.as_str()
            wd = self._workers.get(key)
            if wd is None:
                return
            wd.status = status

