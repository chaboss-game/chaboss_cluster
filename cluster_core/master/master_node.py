"""
Логика мастера: подключение к воркерам, HealthStream (heartbeat),
автопереподключение при обрыве с экспоненциальным backoff.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
import hashlib
from typing import Dict, List

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
GET_STATUS_TIMEOUT_S = 15.0  # таймаут GetStatus при подключении/переподключении (медленная сеть или загрузка воркера)


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
        os_name=desc.resources.os_name or None,
        os_version=desc.resources.os_version or None,
    )
    worker_id = WorkerId(host=desc.id.host, port=desc.id.port)
    return WorkerDescriptor(
        worker_id=worker_id,
        status=WorkerStatus.ONLINE,
        resources=resources,
        token_fingerprint=desc.token_fingerprint or None,
        token_status=desc.token_status or None,
    )


def _token_fingerprint(token: str | None) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12] if token else ""


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
        self._desired_worker_configs: Dict[str, WorkerConfig] = {}  # key -> WorkerConfig, обновляется из UI
        self._stop_event = threading.Event()
        self._last_loaded_model_id: str | None = None

    def start_background(self) -> None:
        """Старт: первичное подключение к воркерам и поток HealthStream для каждого из конфига."""
        n = len(self._cfg.workers)
        if n == 0:
            logger.warning(
                "В конфиге мастера нет воркеров (workers: []). "
                "Добавьте воркеров в config/master.yaml или в UI нажмите «Применить конфиг на мастере»."
            )
        with self._lock:
            self._desired_worker_configs = {f"{w.host}:{w.port}": w for w in self._cfg.workers}
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
        logger.info("Started health stream threads for %d workers from config", n)

    def stop(self) -> None:
        self._stop_event.set()

    def _get_worker_config(self, key: str) -> WorkerConfig | None:
        with self._lock:
            return self._desired_worker_configs.get(key) or next(
                (w for w in self._cfg.workers if f"{w.host}:{w.port}" == key), None
            )

    def _connect_worker(self, w_cfg: WorkerConfig) -> None:
        target = f"{w_cfg.host}:{w_cfg.port}"
        key = target
        logger.info("Connecting to worker %s", target)
        channel = grpc.insecure_channel(target)
        stub = cluster_pb2_grpc.WorkerServiceStub(channel)

        worker_id_pb = cluster_pb2.WorkerId(host=w_cfg.host, port=w_cfg.port)
        try:
            desc = stub.GetStatus(worker_id_pb, timeout=GET_STATUS_TIMEOUT_S)
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
        expected = _token_fingerprint(w_cfg.auth_token)
        got = wd.token_fingerprint or ""
        if not expected and not got:
            wd.token_status = "NONE"
        elif expected == got:
            wd.token_status = "OK"
        else:
            wd.token_status = "MISMATCH"
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
            desc = stub.GetStatus(worker_id_pb, timeout=GET_STATUS_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reconnect GetStatus failed for %s: %s", target, exc)
            channel.close()
            return False

        wd = _descriptor_from_proto(desc)
        expected = _token_fingerprint(w_cfg.auth_token)
        got = wd.token_fingerprint or ""
        if not expected and not got:
            wd.token_status = "NONE"
        elif expected == got:
            wd.token_status = "OK"
        else:
            wd.token_status = "MISMATCH"
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
                if worker_key not in self._desired_worker_configs:
                    return
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

    def update_workers_config(self, workers: List[WorkerConfig]) -> tuple[bool, str]:
        """
        Обновляет список воркеров (из UI): отключает удалённых, подключает новых.
        Без перезапуска мастера.
        """
        with self._lock:
            new_desired = {f"{w.host}:{w.port}": w for w in workers}
            to_remove = [k for k in self._channels if k not in new_desired]
            to_add = [(k, w_cfg) for k, w_cfg in new_desired.items() if k not in self._stubs]
            self._desired_worker_configs = new_desired

        for key in to_remove:
            with self._lock:
                ch = self._channels.pop(key, None)
                self._stubs.pop(key, None)
            if ch is not None:
                try:
                    ch.close()
                except Exception:  # noqa: S110
                    pass
            wid = _parse_worker_id(key)
            self._registry.set_status(wid, WorkerStatus.OFFLINE)
            logger.info("Worker %s removed from config", key)

        for key, w_cfg in to_add:
            self._connect_worker(w_cfg)
            with self._lock:
                if key not in self._stubs:
                    continue
            t = threading.Thread(
                target=self._health_stream_loop,
                args=(key, w_cfg),
                daemon=True,
                name=f"health-{key}",
            )
            t.start()
            logger.info("Started health stream for new worker %s", key)

        return True, ""

    def unload_model(self, model_id: str | None = None) -> tuple[bool, str]:
        """
        Выгружает шарды с воркеров (освобождение VRAM).
        model_id: конкретная модель или None/"" — выгрузить текущую загруженную модель.
        Возвращает (ok, error_message).
        """
        with self._lock:
            worker_keys = list(self._stubs.keys())
        if not worker_keys:
            return True, ""
        mid = (model_id or "").strip() or self._last_loaded_model_id
        if not mid:
            return True, ""
        errors: list[str] = []
        for i, key in enumerate(worker_keys):
            stub = self._stubs.get(key)
            if stub is None:
                continue
            req = cluster_pb2.UnloadShardRequest(model_id=mid, shard_id=str(i))
            try:
                resp = stub.UnloadShard(req, timeout=30.0)
                if not resp.ok:
                    errors.append(f"{key}: {resp.error}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{key}: {e}")
        if mid == self._last_loaded_model_id:
            self._last_loaded_model_id = None
        if errors:
            return False, "; ".join(errors)
        return True, ""

    def load_model(self, hf_model_id: str) -> tuple[bool, str]:
        """
        Загружает модель с HF (если нет в кэше — скачивает), дробит на чанки
        и рассылает воркерам через InitShard (холодная загрузка).
        Перед загрузкой выгружает текущую модель с воркеров (освобождение VRAM).
        Возвращает (ok, error_message).
        """
        from cluster_core.master.model_loader import prepare_shards

        self.unload_model(None)

        with self._lock:
            worker_keys = list(self._stubs.keys())
        if not worker_keys:
            return False, "Нет подключённых воркеров"

        # Запрет работы с моделью при несоответствии токенов
        bad = [k for k, wd in self._registry.all().items() if (wd.token_status or "") == "MISMATCH"]
        if bad:
            return False, "Несоответствие auth_token у воркеров: " + ", ".join(sorted(bad))

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
