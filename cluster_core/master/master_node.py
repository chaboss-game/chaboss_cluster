"""
Логика мастера: подключение к воркерам, HealthStream (heartbeat),
автопереподключение при обрыве с экспоненциальным backoff.
"""
from __future__ import annotations

import logging
import queue
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
GET_STATUS_TIMEOUT_S = 30.0  # таймаут GetStatus при подключении/переподключении (медленная сеть, холодный старт Windows)
# InitShard: загрузка с HF и загрузка шарда на воркере может занимать десятки минут.
# Явно задаём большой таймаут (7 дней); None в gRPC может подменяться дефолтом.
INIT_SHARD_TIMEOUT_S = 86400 * 7

# Keepalive для долгоживущих каналов к воркерам (снижает реконнекты из-за NAT/файрвола).
WORKER_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 15_000),  # пинг каждые 15 с
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
]


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
        self._last_loaded_shard_ids: List[str] = []
        # Число подряд неудачных переподключений; RECONNECTING показываем только после N подряд.
        self._reconnect_failures: Dict[str, int] = {}
        self._reconnect_failures_before_status = 2

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
        channel = grpc.insecure_channel(target, options=WORKER_CHANNEL_OPTIONS)
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
        logger.info(
            "Worker %s token check: expected=%s got=%s status=%s",
            key,
            expected or "<none>",
            got or "<none>",
            wd.token_status,
        )
        self._registry.upsert(wd)
        with self._lock:
            self._channels[key] = channel
            self._stubs[key] = stub
        logger.info("Worker %s registered as ONLINE", key)

    def _reconnect_worker(self, w_cfg: WorkerConfig) -> bool:
        """Создать новый channel/stub, GetStatus, обновить реестр и _channels/_stubs."""
        target = f"{w_cfg.host}:{w_cfg.port}"
        key = target
        channel = grpc.insecure_channel(target, options=WORKER_CHANNEL_OPTIONS)
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
        logger.info(
            "Worker %s token check (reconnect): expected=%s got=%s status=%s",
            key,
            expected or "<none>",
            got or "<none>",
            wd.token_status,
        )
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

            # Всегда берём актуальный конфиг (включая auth_token) из desired/config.
            current_cfg = self._get_worker_config(worker_key) or w_cfg

            if stub is None:
                if self._reconnect_worker(current_cfg):
                    backoff_s = RECONNECT_BACKOFF_INITIAL_S
                    self._reconnect_failures[worker_key] = 0
                else:
                    n = self._reconnect_failures.get(worker_key, 0) + 1
                    self._reconnect_failures[worker_key] = n
                    if n >= self._reconnect_failures_before_status:
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
                # Долгоживущий поток; дедлайн не ставим — иначе флаппинг статуса.
                self._registry.set_status(wid, WorkerStatus.ONLINE)
                for _ in stub.HealthStream(ping_iterator()):
                    pass
            except grpc.RpcError as e:
                logger.warning(
                    "HealthStream обрыв (мастер) %s: code=%s details=%s",
                    worker_key, e.code(), e.details() or "",
                )
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
            # RECONNECTING показываем только когда не удаётся переподключиться (см. stub is None).
            # При обрыве стрима просто переподключаемся без смены статуса — без флаппинга в UI.
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, RECONNECT_BACKOFF_MAX_S)

    def update_workers_config(self, workers: List[WorkerConfig]) -> tuple[bool, str]:
        """
        Обновляет список воркеров (из UI): отключает удалённых, подключает новых.
        Без перезапуска мастера.
        """
        with self._lock:
            prev_desired = dict(self._desired_worker_configs)
            new_desired: Dict[str, WorkerConfig] = {}
            for w in workers:
                key = f"{w.host}:{w.port}"
                # Защита от случайного "сброса" токена: если UI прислал пустой auth_token,
                # сохраняем уже известный токен для этого воркера.
                if (w.auth_token is None or w.auth_token == "") and key in prev_desired:
                    w = WorkerConfig(host=w.host, port=w.port, auth_token=prev_desired[key].auth_token)
                new_desired[key] = w
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
        from cluster_core.master.model_loader import prepare_shard_keys, prepare_layer_ranges

        self.unload_model(None)

        with self._lock:
            worker_keys = list(self._stubs.keys())
        if not worker_keys:
            return False, "Нет подключённых воркеров"

        # Запрет работы с моделью при несоответствии токенов
        bad = [k for k, wd in self._registry.all().items() if (wd.token_status or "") == "MISMATCH"]
        if bad:
            return False, "Несоответствие auth_token у воркеров: " + ", ".join(sorted(bad))

        mode = getattr(self._cfg, "model_load_mode", "fit_in_cluster") or "fit_in_cluster"
        shard_keys_list: List[List[str]] | None = None
        shard_id_list: List[str] = []

        if mode == "streaming_chunks":
            logger.info("Загрузка модели %s (streaming_chunks): планирование диапазонов слоёв, воркеров=%d", hf_model_id, len(worker_keys))
            try:
                ranges = prepare_layer_ranges(hf_model_id, len(worker_keys))
            except Exception as e:  # noqa: BLE001
                logger.exception("prepare_layer_ranges failed for %s: %s", hf_model_id, e)
                return False, str(e)
            # shard_id кодируем как "start-end" (end не включительно)
            shard_id_list = [f"{a}-{b}" for a, b in ranges]
            # если воркеров больше, чем диапазонов (слоёв мало) — оставшиеся пропускаем
            worker_keys = worker_keys[: len(shard_id_list)]
            logger.info("Streaming plan готов: %s", shard_id_list)
        else:
            logger.info(
                "Загрузка модели %s (fit_in_cluster): подготовка меток шардов на мастере (HF -> кэш, разбивка по слоям), воркеров=%d",
                hf_model_id, len(worker_keys),
            )
            try:
                shard_keys_list = prepare_shard_keys(hf_model_id, len(worker_keys))
            except Exception as e:  # noqa: BLE001
                logger.exception("prepare_shard_keys failed for %s: %s", hf_model_id, e)
                return False, str(e)
            shard_id_list = [str(i) for i in range(len(worker_keys))]

        logger.info("Команда воркерам подготовиться к модели %s: %s", hf_model_id, worker_keys)
        errors: list[str] = []
        for i, key in enumerate(worker_keys):
            stub = self._stubs.get(key)
            if stub is None:
                logger.warning("Воркер %s: нет соединения, пропуск шарда %d", key, i)
                errors.append(f"{key}: нет соединения")
                continue
            shard_id = shard_id_list[i] if i < len(shard_id_list) else str(i)
            shard_keys = shard_keys_list[i] if (shard_keys_list is not None and i < len(shard_keys_list)) else []
            logger.info("Команда воркеру %s: скачать с HF и загрузить shard=%d (keys=%d)", key, i, len(shard_keys))
            wid = _parse_worker_id(key)
            req = cluster_pb2.InitShardRequest(
                spec=cluster_pb2.ShardSpec(
                    model_id=hf_model_id,
                    shard_id=shard_id,
                    backend="baseline",
                ),
                weight_source="hf_stream" if mode == "streaming_chunks" else "hf",
                hf_model_name=hf_model_id,
                shard_keys=shard_keys,
            )
            try:
                resp = stub.InitShard(req, timeout=INIT_SHARD_TIMEOUT_S)
                if not resp.ok:
                    logger.warning("Воркер %s: ошибка загрузки шарда: %s", key, resp.error)
                    errors.append(f"{key}: {resp.error}")
                else:
                    logger.info("Воркер %s: шард загружен успешно", key)
            except Exception as e:  # noqa: BLE001
                logger.warning("Воркер %s: исключение при загрузке шарда: %s", key, e, exc_info=True)
                errors.append(f"{key}: {e}")

        if errors:
            return False, "; ".join(errors)
        self._last_loaded_model_id = hf_model_id
        self._last_loaded_shard_ids = shard_id_list
        return True, ""

    def load_model_with_progress(
        self,
        hf_model_id: str,
        progress_queue: "queue.Queue[cluster_pb2.LoadModelProgressEvent]",
    ) -> None:
        """
        То же, что load_model, но кладёт в progress_queue события LoadModelProgressEvent
        (прогресс мастера и воркеров). В конце кладёт событие с done=True, ok=..., error=...
        """
        from cluster_core.master.model_loader import prepare_shard_keys, prepare_layer_ranges

        def make_master_progress(stage: str, percent: int, bytes_done: int = 0, bytes_total: int = 0, current_file: str = "") -> cluster_pb2.LoadProgress:
            return cluster_pb2.LoadProgress(
                stage=stage,
                percent=percent,
                bytes_downloaded=bytes_done,
                bytes_total=bytes_total,
                current_file=current_file or "",
            )

        def make_event(master: cluster_pb2.LoadProgress | None, workers: Dict[str, cluster_pb2.LoadProgress], done: bool = False, ok: bool = False, error: str = "") -> cluster_pb2.LoadModelProgressEvent:
            ev = cluster_pb2.LoadModelProgressEvent(done=done, ok=ok, error=error or "")
            if master is not None:
                ev.master.CopyFrom(master)
            for k, v in workers.items():
                ev.workers[k].CopyFrom(v)
            return ev

        try:
            # Сразу отправляем первое событие, чтобы GUI не висел в ожидании (unload может быть долгим).
            progress_queue.put(make_event(make_master_progress("unload", 0), {}))
            self.unload_model(None)
            with self._lock:
                worker_keys = list(self._stubs.keys())
                stubs_snapshot = dict(self._stubs)
            if not worker_keys:
                progress_queue.put(make_event(None, {}, done=True, ok=False, error="Нет подключённых воркеров"))
                return
            bad = [k for k, wd in self._registry.all().items() if (wd.token_status or "") == "MISMATCH"]
            if bad:
                progress_queue.put(make_event(None, {}, done=True, ok=False, error="Несоответствие auth_token у воркеров: " + ", ".join(sorted(bad))))
                return

            # Сразу эмитим начальное событие, чтобы в GUI отобразился прогресс-бар.
            progress_queue.put(make_event(make_master_progress("download", 0), {}))

            def on_master_progress(percent: float, bytes_done: int, bytes_total: int, current_file: str) -> None:
                progress_queue.put(make_event(
                    make_master_progress("download", min(100, int(percent)), bytes_done, bytes_total, current_file),
                    {},
                ))

            mode = getattr(self._cfg, "model_load_mode", "fit_in_cluster") or "fit_in_cluster"
            shard_keys_list: List[List[str]] | None = None
            shard_id_list: List[str] = []
            if mode == "streaming_chunks":
                # Для streaming_chunks мастер скачивает только ради прогресса/плана; веса целиком в RAM не грузим.
                ranges = prepare_layer_ranges(hf_model_id, len(worker_keys), progress_callback=on_master_progress)
                shard_id_list = [f"{a}-{b}" for a, b in ranges]
                worker_keys = worker_keys[: len(shard_id_list)]
                with self._lock:
                    stubs_snapshot = {k: stubs_snapshot[k] for k in worker_keys if k in stubs_snapshot}
                progress_queue.put(make_event(make_master_progress("plan", 100), {}))
            else:
                shard_keys_list = prepare_shard_keys(hf_model_id, len(worker_keys), progress_callback=on_master_progress)
                shard_id_list = [str(i) for i in range(len(worker_keys))]
                progress_queue.put(make_event(make_master_progress("sharding", 100), {}))

            errors: list[str] = []
            workers_progress: Dict[str, cluster_pb2.LoadProgress] = {}
            for i, key in enumerate(worker_keys):
                stub = stubs_snapshot.get(key)
                if stub is None:
                    errors.append(f"{key}: нет соединения")
                    continue
                shard_id = shard_id_list[i] if i < len(shard_id_list) else str(i)
                shard_keys = shard_keys_list[i] if (shard_keys_list is not None and i < len(shard_keys_list)) else []
                wid = _parse_worker_id(key)
                req = cluster_pb2.InitShardRequest(
                    spec=cluster_pb2.ShardSpec(model_id=hf_model_id, shard_id=shard_id, backend="baseline"),
                    weight_source="hf_stream" if mode == "streaming_chunks" else "hf",
                    hf_model_name=hf_model_id,
                    shard_keys=shard_keys,
                )
                result: list = [None]
                # Сразу показываем воркер в GUI с 0%, чтобы бар появился
                workers_progress[key] = cluster_pb2.LoadProgress(stage="download", percent=0)
                progress_queue.put(make_event(make_master_progress("workers", 100), dict(workers_progress)))
                def do_init() -> None:
                    result[0] = stub.InitShard(req, timeout=INIT_SHARD_TIMEOUT_S)
                t = threading.Thread(target=do_init)
                t.start()
                worker_id_pb = cluster_pb2.WorkerId(host=wid.host, port=wid.port)
                while t.is_alive():
                    time.sleep(0.25)
                    try:
                        r = stub.GetLoadProgress(worker_id_pb, timeout=2.0)
                        if r.progress:
                            workers_progress[key] = r.progress
                            progress_queue.put(make_event(make_master_progress("workers", 100), dict(workers_progress)))
                    except Exception:  # noqa: S110
                        pass
                t.join()
                if result[0] and not result[0].ok:
                    errors.append(f"{key}: {result[0].error}")
                elif result[0] is None:
                    errors.append(f"{key}: нет ответа")
                workers_progress[key] = cluster_pb2.LoadProgress(stage="done", percent=100)
                progress_queue.put(make_event(make_master_progress("workers", 100), dict(workers_progress)))

            if errors:
                progress_queue.put(make_event(None, {}, done=True, ok=False, error="; ".join(errors)))
                return
            self._last_loaded_model_id = hf_model_id
            self._last_loaded_shard_ids = shard_id_list
            progress_queue.put(make_event(None, {}, done=True, ok=True, error=""))
        except Exception as e:  # noqa: BLE001
            logger.exception("load_model_with_progress failed: %s", e)
            progress_queue.put(make_event(None, {}, done=True, ok=False, error=str(e)))

    def get_last_loaded_model_id(self) -> str | None:
        return self._last_loaded_model_id

    def remote_update_workers(
        self,
        restart_gui: bool,
        start_worker: bool,
        git_remote: str = "origin",
        git_branch: str = "",
    ) -> tuple[bool, str, list[dict]]:
        """
        Рассылает воркерам команду обновиться (git pull) и опционально перезапустить GUI.
        Возвращает (ok, error, results[]), где results содержит worker/ok/error/output.
        """
        with self._lock:
            stubs_snapshot = dict(self._stubs)
            worker_keys = list(stubs_snapshot.keys())
        if not worker_keys:
            return False, "Нет подключённых воркеров", []

        results: list[dict] = []
        errors: list[str] = []
        for key in worker_keys:
            stub = stubs_snapshot.get(key)
            if stub is None:
                results.append({"worker": key, "ok": False, "error": "нет соединения", "output": ""})
                errors.append(f"{key}: нет соединения")
                continue
            try:
                r = stub.RemoteUpdate(
                    cluster_pb2.RemoteUpdateRequest(
                        restart_gui=restart_gui,
                        start_worker=start_worker,
                        git_remote=git_remote or "origin",
                        git_branch=git_branch or "",
                    ),
                    timeout=240.0,
                )
                results.append({"worker": key, "ok": bool(r.ok), "error": r.error or "", "output": r.output or ""})
                if not r.ok:
                    errors.append(f"{key}: {r.error or 'ошибка'}")
            except Exception as e:  # noqa: BLE001
                results.append({"worker": key, "ok": False, "error": str(e), "output": ""})
                errors.append(f"{key}: {e}")

        ok = len(errors) == 0
        return ok, "; ".join(errors), results

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
        shard_ids = list(self._last_loaded_shard_ids) if self._last_loaded_shard_ids else [str(i) for i in range(len(worker_keys))]

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
                shard_id=shard_ids[i] if i < len(shard_ids) else str(i),
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
