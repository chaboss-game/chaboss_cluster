"""
Логика мастера: подключение к воркерам, HealthStream (heartbeat),
автопереподключение при обрыве с экспоненциальным backoff.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict

import grpc

from cluster_core.common.config import MasterConfig, WorkerConfig
from cluster_core.common.tensor_io import payload_to_tensor, tensor_to_payload
from cluster_core.common.types import GpuInfo, ResourceInfo, WorkerDescriptor, WorkerId, WorkerStatus
from cluster_core.master.worker_registry import WorkerRegistry
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc


logger = logging.getLogger("master.node")

HEARTBEAT_INTERVAL_S = 2.0
RECONNECT_BACKOFF_INITIAL_S = 1.0
RECONNECT_BACKOFF_MAX_S = 30.0


def _parse_worker_id(key: str) -> WorkerId:
    host, port_str = key.rsplit(":", 1)
    return WorkerId(host=host, port=int(port_str))


def _descriptor_from_proto(desc: cluster_pb2.WorkerDescriptor) -> WorkerDescriptor:
    gpus = [
        GpuInfo(
            index=g.index,
            name=g.name,
            total_vram_mb=g.total_vram_mb,
            compute_capability=g.compute_capability or None,
            backend=g.backend or "cuda",
        )
        for g in desc.resources.gpus
    ]
    resources = ResourceInfo(
        cpu_cores=desc.resources.cpu_cores,
        ram_total_mb=desc.resources.ram_total_mb,
        ram_available_mb=desc.resources.ram_available_mb,
        gpus=gpus,
        torch_version=desc.resources.torch_version or None,
        cuda_version=desc.resources.cuda_version or None,
        rocm_version=desc.resources.rocm_version or None,
    )
    worker_id = WorkerId(host=desc.id.host, port=desc.id.port)
    return WorkerDescriptor(
        worker_id=worker_id,
        status=WorkerStatus.ONLINE,
        resources=resources,
    )


class MasterNode:
    """
    Подключение к воркерам из конфига, первичный GetStatus,
    затем для каждого воркера — долгоживущий HealthStream (bidi).
    При обрыве: статус RECONNECTING, backoff, переподключение и при успехе — снова ONLINE.
    """

    def __init__(self, cfg: MasterConfig, registry: WorkerRegistry) -> None:
        self._cfg = cfg
        self._registry = registry
        self._lock = threading.RLock()
        self._channels: Dict[str, grpc.Channel] = {}
        self._stubs: Dict[str, cluster_pb2_grpc.WorkerServiceStub] = {}
        self._stop_event = threading.Event()
        self._last_loaded_model_id: str | None = None

    def start_background(self) -> None:
        """Старт: первичное подключение к воркерам и поток HealthStream для каждого из конфига."""
        for w_cfg in self._cfg.workers:
            self._connect_worker(w_cfg)

        for w_cfg in self._cfg.workers:
            key = f"{w_cfg.host}:{w_cfg.port}"
            t = threading.Thread(
                target=self._health_stream_loop,
                args=(key, w_cfg),
                daemon=True,
                name=f"health-{key}",
            )
            t.start()
        logger.info("Started health stream threads for %d workers", len(self._cfg.workers))

    def stop(self) -> None:
        self._stop_event.set()

    def _get_worker_config(self, key: str) -> WorkerConfig | None:
        wid = _parse_worker_id(key)
        for w in self._cfg.workers:
            if w.host == wid.host and w.port == wid.port:
                return w
        return None

    def _connect_worker(self, w_cfg: WorkerConfig) -> None:
        target = f"{w_cfg.host}:{w_cfg.port}"
        key = target
        logger.info("Connecting to worker %s", target)
        channel = grpc.insecure_channel(target)
        stub = cluster_pb2_grpc.WorkerServiceStub(channel)

        worker_id_pb = cluster_pb2.WorkerId(host=w_cfg.host, port=w_cfg.port)
        try:
            desc = stub.GetStatus(worker_id_pb, timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to get status from worker %s: %s", target, exc)
            channel.close()
            wd = WorkerDescriptor(
                worker_id=WorkerId(host=w_cfg.host, port=w_cfg.port),
                status=WorkerStatus.OFFLINE,
                resources=ResourceInfo(cpu_cores=0, ram_total_mb=0, ram_available_mb=0),
            )
            self._registry.upsert(wd)
            return

        wd = _descriptor_from_proto(desc)
        self._registry.upsert(wd)
        with self._lock:
            self._channels[key] = channel
            self._stubs[key] = stub
        logger.info("Worker %s registered as ONLINE", key)

    def _reconnect_worker(self, w_cfg: WorkerConfig) -> bool:
        """Создать новый channel/stub, GetStatus, обновить реестр и _channels/_stubs."""
        target = f"{w_cfg.host}:{w_cfg.port}"
        key = target
        channel = grpc.insecure_channel(target)
        stub = cluster_pb2_grpc.WorkerServiceStub(channel)
        worker_id_pb = cluster_pb2.WorkerId(host=w_cfg.host, port=w_cfg.port)
        try:
            desc = stub.GetStatus(worker_id_pb, timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reconnect GetStatus failed for %s: %s", target, exc)
            channel.close()
            return False

        wd = _descriptor_from_proto(desc)
        self._registry.upsert(wd)
        with self._lock:
            old_ch = self._channels.pop(key, None)
            if old_ch is not None:
                try:
                    old_ch.close()
                except Exception:  # noqa: S110
                    pass
            self._channels[key] = channel
            self._stubs[key] = stub
        logger.info("Worker %s reconnected, ONLINE", key)
        return True

    def _health_stream_loop(self, worker_key: str, w_cfg: WorkerConfig) -> None:
        wid = _parse_worker_id(worker_key)
        backoff_s = RECONNECT_BACKOFF_INITIAL_S

        while not self._stop_event.is_set():
            with self._lock:
                stub = self._stubs.get(worker_key)

            if stub is None:
                if self._reconnect_worker(w_cfg):
                    backoff_s = RECONNECT_BACKOFF_INITIAL_S
                else:
                    self._registry.set_status(wid, WorkerStatus.RECONNECTING)
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, RECONNECT_BACKOFF_MAX_S)
                continue

            def ping_iterator():
                while not self._stop_event.is_set():
                    time.sleep(HEARTBEAT_INTERVAL_S)
                    if self._stop_event.is_set():
                        return
                    yield cluster_pb2.HealthPing(
                        id=cluster_pb2.WorkerId(host=wid.host, port=wid.port),
                        nonce=str(uuid.uuid4()),
                    )

            try:
                for _ in stub.HealthStream(ping_iterator(), timeout=30):
                    self._registry.set_status(wid, WorkerStatus.ONLINE)
            except grpc.RpcError as e:
                logger.warning("HealthStream RpcError for %s: %s", worker_key, e)
            except Exception as e:  # noqa: BLE001
                logger.warning("HealthStream failed for %s: %s", worker_key, e)

            with self._lock:
                ch = self._channels.pop(worker_key, None)
                self._stubs.pop(worker_key, None)
                if ch is not None:
                    try:
                        ch.close()
                    except Exception:  # noqa: S110
                        pass

            self._registry.set_status(wid, WorkerStatus.RECONNECTING)
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, RECONNECT_BACKOFF_MAX_S)

    def load_model(self, hf_model_id: str) -> tuple[bool, str]:
        """
        Загружает модель с HF (если нет в кэше — скачивает), дробит на чанки
        и рассылает воркерам через InitShard (холодная загрузка).
        Возвращает (ok, error_message).
        """
        from cluster_core.master.model_loader import prepare_shards

        with self._lock:
            worker_keys = list(self._stubs.keys())
        if not worker_keys:
            return False, "Нет подключённых воркеров"

        try:
            shards = prepare_shards(hf_model_id, len(worker_keys))
        except Exception as e:  # noqa: BLE001
            logger.exception("prepare_shards failed for %s", hf_model_id)
            return False, str(e)

        errors: list[str] = []
        for i, key in enumerate(worker_keys):
            stub = self._stubs.get(key)
            if stub is None:
                errors.append(f"{key}: нет соединения")
                continue
            wid = _parse_worker_id(key)
            req = cluster_pb2.InitShardRequest(
                spec=cluster_pb2.ShardSpec(
                    model_id=hf_model_id,
                    shard_id=str(i),
                    backend="baseline",
                ),
                weight_source="inline_blob",
                inline_blob=shards[i] if i < len(shards) else b"",
            )
            try:
                resp = stub.InitShard(req, timeout=120.0)
                if not resp.ok:
                    errors.append(f"{key}: {resp.error}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{key}: {e}")

        if errors:
            return False, "; ".join(errors)
        self._last_loaded_model_id = hf_model_id
        return True, ""

    def get_last_loaded_model_id(self) -> str | None:
        return self._last_loaded_model_id

    def run_pipeline(self, input_tensor: "torch.Tensor") -> "torch.Tensor":
        """
        Прогоняет тензор по цепочке воркеров (RunStage): W1 -> W2 -> ... -> выход.
        Порядок воркеров совпадает с порядком при load_model.
        """
        with self._lock:
            worker_keys = list(self._stubs.keys())
        if not worker_keys:
            raise RuntimeError("Нет подключённых воркеров для pipeline")

        current = input_tensor
        request_id = str(uuid.uuid4())
        model_id = self._last_loaded_model_id or ""

        for i, key in enumerate(worker_keys):
            stub = self._stubs.get(key)
            if stub is None:
                raise RuntimeError(f"Воркер {key} недоступен")
            payload = tensor_to_payload(current)
            req = cluster_pb2.StageRequest(
                request_id=request_id,
                tensor=payload,
                is_last=True,
                model_id=model_id,
                shard_id=str(i),
            )
            response_iter = stub.RunStage(iter([req]), timeout=60.0)
            resp = next(response_iter, None)
            if resp is None:
                raise RuntimeError(f"Воркер {key} не вернул ответ")
            if resp.error:
                raise RuntimeError(f"Воркер {key}: {resp.error}")
            if not resp.tensor or not resp.tensor.data:
                raise RuntimeError(f"Воркер {key}: пустой тензор в ответе")
            current = payload_to_tensor(resp.tensor, device="cpu")

        return current
