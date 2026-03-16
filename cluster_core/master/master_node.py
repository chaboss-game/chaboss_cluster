from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, Iterable

import grpc

from cluster_core.common.config import MasterConfig, WorkerConfig
from cluster_core.common.types import ResourceInfo, WorkerDescriptor, WorkerId, WorkerStatus
from cluster_core.master.worker_registry import WorkerRegistry
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc


logger = logging.getLogger("master.node")


class MasterNode:
    """
    Основная логика мастера:
    - подключение к воркерам из конфига;
    - первичный GetStatus;
    - поддержка HealthStream (heartbeat + реконнект, позже).
    """

    def __init__(self, cfg: MasterConfig, registry: WorkerRegistry) -> None:
        self._cfg = cfg
        self._registry = registry
        self._channels: Dict[str, grpc.Channel] = {}
        self._stubs: Dict[str, cluster_pb2_grpc.WorkerServiceStub] = {}
        self._stop_event = threading.Event()

    def start_background(self) -> None:
        """
        Старт фоновых потоков для управления воркерами.
        """
        t = threading.Thread(target=self._bootstrap_workers, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop_event.set()

    # Internal helpers

    def _bootstrap_workers(self) -> None:
        """
        Подключаемся к воркерам из конфига, получаем их статус
        и запускаем heartbeat‑цикл.
        """
        for w_cfg in self._cfg.workers:
            self._connect_worker(w_cfg)

        # Простой heartbeat‑цикл (пока без полноценного HealthStream).
        while not self._stop_event.is_set():
            self._refresh_workers_status()
            time.sleep(5.0)

    def _connect_worker(self, w_cfg: WorkerConfig) -> None:
        target = f"{w_cfg.host}:{w_cfg.port}"
        logger.info("Connecting to worker %s", target)
        channel = grpc.insecure_channel(target)
        stub = cluster_pb2_grpc.WorkerServiceStub(channel)

        worker_id = cluster_pb2.WorkerId(host=w_cfg.host, port=w_cfg.port)
        try:
            desc = stub.GetStatus(worker_id, timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to get status from worker %s: %s", target, exc)
            wd = WorkerDescriptor(
                worker_id=WorkerId(host=w_cfg.host, port=w_cfg.port),
                status=WorkerStatus.OFFLINE,
                resources=ResourceInfo(cpu_cores=0, ram_total_mb=0, ram_available_mb=0),
            )
            self._registry.upsert(wd)
            return

        resources = ResourceInfo(
            cpu_cores=desc.resources.cpu_cores,
            ram_total_mb=desc.resources.ram_total_mb,
            ram_available_mb=desc.resources.ram_available_mb,
            gpus=[],
            torch_version=desc.resources.torch_version or None,
            cuda_version=desc.resources.cuda_version or None,
            rocm_version=desc.resources.rocm_version or None,
        )
        worker_id_obj = WorkerId(host=desc.id.host, port=desc.id.port)
        wd = WorkerDescriptor(
            worker_id=worker_id_obj,
            status=WorkerStatus.ONLINE,
            resources=resources,
        )
        self._registry.upsert(wd)
        self._channels[worker_id_obj.as_str()] = channel
        self._stubs[worker_id_obj.as_str()] = stub
        logger.info("Worker %s registered as ONLINE", worker_id_obj.as_str())

    def _refresh_workers_status(self) -> None:
        """
        Периодически получать статус от уже известных воркеров.
        Пока используем GetStatus; позже перейдём на постоянный HealthStream.
        """
        for key, stub in list(self._stubs.items()):
            wid = self._parse_worker_id(key)
            try:
                desc = stub.GetStatus(cluster_pb2.WorkerId(host=wid.host, port=wid.port), timeout=2.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Worker %s appears offline: %s", key, exc)
                self._registry.set_status(wid, WorkerStatus.OFFLINE)
                continue

            self._registry.set_status(wid, WorkerStatus.ONLINE)

    @staticmethod
    def _parse_worker_id(key: str) -> WorkerId:
        host, port_str = key.rsplit(":", 1)
        return WorkerId(host=host, port=int(port_str))

