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
from pathlib import Path
from typing import Any, Dict, List

import grpc

from cluster_core.common.config import MasterConfig, WorkerConfig
from cluster_core.common.tensor_io import payload_to_tensor, tensor_to_payload
from cluster_core.common.types import GpuInfo, ResourceInfo, WorkerDescriptor, WorkerId, WorkerStatus
from cluster_core.master.worker_registry import WorkerRegistry
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc
from cluster_core.common.chat_storage import ChatStorage


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


def _is_likely_gptq_model(hf_model_id: str) -> bool:
    """
    Best-effort детектор GPTQ.
    1) Быстрый эвристический чек по имени модели.
    2) Если доступен transformers, проверка quantization_config.quant_method == gptq.
    """
    mid = (hf_model_id or "").strip().lower()
    if "gptq" in mid:
        return True
    try:
        from transformers import AutoConfig  # type: ignore
        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        qc = getattr(cfg, "quantization_config", None)
        if qc is None:
            return False
        if isinstance(qc, dict):
            qm = str(qc.get("quant_method", "")).lower()
            return qm == "gptq"
        qm = str(getattr(qc, "quant_method", "")).lower()
        return qm == "gptq"
    except Exception:
        return False


def _worker_budget_mb_from_descriptor(wd: WorkerDescriptor | None) -> int:
    if wd is None:
        return 0
    ram = int(getattr(wd.resources, "ram_available_mb", 0) or 0)
    vram = 0
    for g in getattr(wd.resources, "gpus", []) or []:
        try:
            vram += int(getattr(g, "total_vram_mb", 0) or 0)
        except Exception:
            pass
    return ram + vram


def _parse_worker_id(key: str) -> WorkerId:
    host, port_str = key.rsplit(":", 1)
    return WorkerId(host=host, port=int(port_str))


def _endpoint_matches(expected_host: str, expected_port: int, actual_host: str, actual_port: int) -> bool:
    """
    Строгая валидация endpoint воркера.
    На текущем этапе считаем валидным только точное совпадение host:port.
    """
    return (expected_host or "").strip() == (actual_host or "").strip() and int(expected_port) == int(actual_port)


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

    def __init__(self, cfg: MasterConfig, registry: WorkerRegistry, log_buffer: Any = None) -> None:
        self._cfg = cfg
        self._registry = registry
        self._lock = threading.RLock()
        self._channels: Dict[str, grpc.Channel] = {}
        self._stubs: Dict[str, cluster_pb2_grpc.WorkerServiceStub] = {}
        self._desired_worker_configs: Dict[str, WorkerConfig] = {}  # key -> WorkerConfig, обновляется из UI
        self._stop_event = threading.Event()
        self._last_loaded_model_id: str | None = None
        self._last_loaded_shard_ids: List[str] = []
        self._reconnect_failures: Dict[str, int] = {}
        self._reconnect_failures_before_status = 2
        self._log_buffer = log_buffer  # для GetClusterLogs (буфер из run_master)

        # Чат: история и вложения в ./cluster_shared/
        project_root = Path(__file__).resolve().parents[2]
        shared_root = project_root / "cluster_shared"
        self._chat_storage = ChatStorage(shared_root, cache_messages=2000)
        # Защита от повторной пересылки одного сообщения (идемпотентность при ретраях)
        self._chat_forwarded_ids: Dict[str, float] = {}  # message_id -> time
        self._chat_forwarded_max = 500
        self._chat_forwarded_ttl_s = 120

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

        # Сырой ответ воркера в лог для отладки (виден в окне «Лог» GUI).
        logger.info(
            "GetStatus raw response [target=%s]: id.host=%r id.port=%s status=%s token_fingerprint=%s "
            "token_status=%s cpu_cores=%s ram_total_mb=%s ram_available_mb=%s os=%s %s",
            target,
            getattr(desc.id, "host", None),
            getattr(desc.id, "port", None),
            getattr(desc, "status", None),
            getattr(desc, "token_fingerprint", None) or "",
            getattr(desc, "token_status", None) or "",
            getattr(desc.resources, "cpu_cores", None),
            getattr(desc.resources, "ram_total_mb", None),
            getattr(desc.resources, "ram_available_mb", None),
            (getattr(desc.resources, "os_name", None) or "") + " " + (getattr(desc.resources, "os_version", None) or ""),
            "gpus=%s" % len(getattr(desc.resources, "gpus", [])),
        )

        if not _endpoint_matches(w_cfg.host, w_cfg.port, desc.id.host, desc.id.port):
            logger.warning(
                "Endpoint mismatch for %s: got worker id %s:%s; connection rejected",
                target,
                desc.id.host,
                desc.id.port,
            )
            channel.close()
            wd = WorkerDescriptor(
                worker_id=WorkerId(host=w_cfg.host, port=w_cfg.port),
                status=WorkerStatus.OFFLINE,
                resources=ResourceInfo(cpu_cores=0, ram_total_mb=0, ram_available_mb=0),
                token_status="ENDPOINT_MISMATCH",
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

        logger.info(
            "GetStatus raw response (reconnect) [target=%s]: id.host=%r id.port=%s status=%s token_fingerprint=%s",
            target,
            getattr(desc.id, "host", None),
            getattr(desc.id, "port", None),
            getattr(desc, "status", None),
            getattr(desc, "token_fingerprint", None) or "",
        )

        if not _endpoint_matches(w_cfg.host, w_cfg.port, desc.id.host, desc.id.port):
            logger.warning(
                "Reconnect endpoint mismatch for %s: got worker id %s:%s; reconnect rejected",
                target,
                desc.id.host,
                desc.id.port,
            )
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
                stream = stub.HealthStream(ping_iterator())
                got_valid_pong = False
                for pong in stream:
                    if not _endpoint_matches(wid.host, wid.port, pong.id.host, pong.id.port):
                        logger.warning(
                            "HealthStream pong endpoint mismatch for %s: got %s:%s",
                            worker_key,
                            pong.id.host,
                            pong.id.port,
                        )
                        raise RuntimeError("health endpoint mismatch")
                    if not got_valid_pong:
                        got_valid_pong = True
                        logger.info(
                            "HealthPong raw [expected=%s]: id.host=%r id.port=%s status=%s nonce=%s",
                            worker_key,
                            getattr(pong.id, "host", None),
                            getattr(pong.id, "port", None),
                            getattr(pong, "status", None),
                            getattr(pong, "nonce", None) or "",
                        )
                        self._registry.set_status(wid, WorkerStatus.ONLINE)
                if not got_valid_pong:
                    raise RuntimeError("health stream closed without pong")
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
            # После обрыва стрима сразу уходим в RECONNECTING, чтобы не держать ложный ONLINE.
            self._registry.set_status(wid, WorkerStatus.RECONNECTING)
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
        logger.info("unload_model: start (requested model_id=%r)", model_id)
        with self._lock:
            worker_keys = list(self._stubs.keys())
        if not worker_keys:
            return True, ""
        mid = (model_id or "").strip() or self._last_loaded_model_id
        if not mid:
            return True, ""
        logger.info("unload_model: unloading model_id=%s from %d workers", mid, len(worker_keys))
        errors: list[str] = []
        for key in worker_keys:
            stub = self._stubs.get(key)
            if stub is None:
                continue
            # При выгрузке модели нужно удалять ВСЕ шарды данного model_id на воркерах.
            # Иначе для streaming_chunks (где shard_id имеет вид "start-end") можно
            # случайно оставить старые шарды, что приводит к крашам воркера при повторном старте.
            req = cluster_pb2.UnloadShardRequest(model_id=mid, shard_id="")
            try:
                logger.info("unload_model: calling UnloadShard on worker=%s model_id=%s", key, mid)
                resp = stub.UnloadShard(req, timeout=30.0)
                if not resp.ok:
                    errors.append(f"{key}: {resp.error}")
                    logger.warning("unload_model: UnloadShard failed worker=%s model_id=%s error=%s", key, mid, resp.error)
                else:
                    logger.info("unload_model: UnloadShard ok worker=%s model_id=%s", key, mid)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{key}: {e}")
                logger.warning("unload_model: exception worker=%s model_id=%s exc=%s", key, mid, e, exc_info=True)
        if mid == self._last_loaded_model_id:
            self._last_loaded_model_id = None
        if errors:
            return False, "; ".join(errors)
        logger.info("unload_model: done ok model_id=%s", mid)
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
        # Раскладываем шарды сначала на более ёмкие ноды (RAM+VRAM), чтобы снизить риск OOM на слабых узлах.
        reg_snapshot = self._registry.all()
        worker_keys = sorted(
            worker_keys,
            key=lambda k: _worker_budget_mb_from_descriptor(reg_snapshot.get(k)),
            reverse=True,
        )

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

        load_run_id = uuid.uuid4().hex[:8]
        logger.info("load_model_with_progress[%s]: start hf_model_id=%s", load_run_id, hf_model_id)

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
            logger.info("load_model_with_progress[%s]: unloading previous model (if any)", load_run_id)
            self.unload_model(None)
            logger.info("load_model_with_progress[%s]: unload step finished", load_run_id)
            with self._lock:
                worker_keys = list(self._stubs.keys())
                stubs_snapshot = dict(self._stubs)
            if not worker_keys:
                progress_queue.put(make_event(None, {}, done=True, ok=False, error="Нет подключённых воркеров"))
                logger.warning("load_model_with_progress[%s]: no connected workers", load_run_id)
                return
            reg_snapshot = self._registry.all()
            worker_keys = sorted(
                worker_keys,
                key=lambda k: _worker_budget_mb_from_descriptor(reg_snapshot.get(k)),
                reverse=True,
            )
            with self._lock:
                stubs_snapshot = {k: stubs_snapshot[k] for k in worker_keys if k in stubs_snapshot}
            logger.info("load_model_with_progress[%s]: connected workers=%d", load_run_id, len(worker_keys))
            logger.info(
                "load_model_with_progress[%s]: worker order by budget=%s",
                load_run_id,
                [
                    f"{k}({_worker_budget_mb_from_descriptor(reg_snapshot.get(k))}MB)"
                    for k in worker_keys
                ],
            )
            bad = [k for k, wd in self._registry.all().items() if (wd.token_status or "") == "MISMATCH"]
            if bad:
                progress_queue.put(make_event(None, {}, done=True, ok=False, error="Несоответствие auth_token у воркеров: " + ", ".join(sorted(bad))))
                logger.warning("load_model_with_progress[%s]: token mismatch workers=%s", load_run_id, bad)
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
            if mode == "fit_in_cluster" and _is_likely_gptq_model(hf_model_id):
                logger.warning(
                    "load_model_with_progress[%s]: GPTQ model in fit_in_cluster -> auto fallback to streaming_chunks (%s)",
                    load_run_id,
                    hf_model_id,
                )
                mode = "streaming_chunks"
                progress_queue.put(make_event(make_master_progress("auto_fallback_streaming", 100), {}))
            if mode == "streaming_chunks":
                # Для streaming_chunks мастер скачивает только ради прогресса/плана; веса целиком в RAM не грузим.
                ranges = prepare_layer_ranges(hf_model_id, len(worker_keys), progress_callback=on_master_progress)
                shard_id_list = [f"{a}-{b}" for a, b in ranges]
                worker_keys = worker_keys[: len(shard_id_list)]
                with self._lock:
                    stubs_snapshot = {k: stubs_snapshot[k] for k in worker_keys if k in stubs_snapshot}
                progress_queue.put(make_event(make_master_progress("plan", 100), {}))
                logger.info(
                    "load_model_with_progress[%s]: streaming_chunks plan ready shards=%d worker_keys=%d shard_ids=%s",
                    load_run_id,
                    len(shard_id_list),
                    len(worker_keys),
                    shard_id_list,
                )
            else:
                shard_keys_list = prepare_shard_keys(hf_model_id, len(worker_keys), progress_callback=on_master_progress)
                shard_id_list = [str(i) for i in range(len(worker_keys))]
                progress_queue.put(make_event(make_master_progress("sharding", 100), {}))
                logger.info(
                    "load_model_with_progress[%s]: fit_in_cluster sharding ready shards=%d shard_key_counts=%s",
                    load_run_id,
                    len(shard_id_list),
                    [len(x) for x in shard_keys_list],
                )

            errors: list[str] = []
            workers_progress: Dict[str, cluster_pb2.LoadProgress] = {}
            for i, key in enumerate(worker_keys):
                stub = stubs_snapshot.get(key)
                if stub is None:
                    errors.append(f"{key}: нет соединения")
                    logger.warning("load_model_with_progress[%s]: stub missing for worker=%s", load_run_id, key)
                    continue
                shard_id = shard_id_list[i] if i < len(shard_id_list) else str(i)
                shard_keys = shard_keys_list[i] if (shard_keys_list is not None and i < len(shard_keys_list)) else []
                wid = _parse_worker_id(key)
                logger.info(
                    "load_model_with_progress[%s]: worker=%s init shard=%s keys=%d",
                    load_run_id,
                    key,
                    shard_id,
                    len(shard_keys) if shard_keys is not None else 0,
                )
                req = cluster_pb2.InitShardRequest(
                    spec=cluster_pb2.ShardSpec(model_id=hf_model_id, shard_id=shard_id, backend="baseline"),
                    weight_source="hf_stream" if mode == "streaming_chunks" else "hf",
                    hf_model_name=hf_model_id,
                    shard_keys=shard_keys,
                )
                result: list = [None, None]  # [InitShardResponse, exception]
                workers_progress[key] = cluster_pb2.LoadProgress(stage="download", percent=0)
                progress_queue.put(make_event(make_master_progress("workers", 100), dict(workers_progress)))

                def do_init() -> None:
                    try:
                        logger.info("load_model_with_progress[%s]: calling InitShard RPC worker=%s shard=%s", load_run_id, key, shard_id)
                        result[0] = stub.InitShard(req, timeout=INIT_SHARD_TIMEOUT_S)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "load_model_with_progress[%s]: InitShard RPC exception worker=%s shard=%s exc=%s",
                            load_run_id,
                            key,
                            shard_id,
                            e,
                            exc_info=True,
                        )
                        result[1] = e

                t = threading.Thread(target=do_init)
                t.start()
                worker_id_pb = cluster_pb2.WorkerId(host=wid.host, port=wid.port)
                last_logged = {"percent": -1, "stage": "", "current_file": ""}
                while t.is_alive():
                    time.sleep(0.25)
                    try:
                        r = stub.GetLoadProgress(worker_id_pb, timeout=2.0)
                        if r.progress:
                            workers_progress[key] = r.progress
                            p_int = int(r.progress.percent or 0)
                            stage = r.progress.stage or ""
                            cur_file = r.progress.current_file or ""
                            # Логируем только при существенных изменениях, чтобы не утопить буфер.
                            if (
                                p_int != last_logged["percent"]
                                and (p_int >= 100 or (p_int - last_logged["percent"] >= 10))
                            ) or (stage != last_logged["stage"] or cur_file != last_logged["current_file"]):
                                logger.info(
                                    "load_model_with_progress[%s]: progress worker=%s shard=%s stage=%s percent=%d file=%s",
                                    load_run_id,
                                    key,
                                    shard_id,
                                    stage,
                                    p_int,
                                    cur_file,
                                )
                                last_logged = {"percent": p_int, "stage": stage, "current_file": cur_file}
                            progress_queue.put(make_event(make_master_progress("workers", 100), dict(workers_progress)))
                    except Exception:  # noqa: S110
                        pass
                t.join()
                if result[0] and not result[0].ok:
                    errors.append(f"{key}: {result[0].error}")
                    logger.warning(
                        "load_model_with_progress[%s]: InitShard finished ok=False worker=%s shard=%s error=%s",
                        load_run_id,
                        key,
                        shard_id,
                        result[0].error,
                    )
                elif result[0] is None:
                    exc = result[1]
                    if exc is None:
                        err_detail = "нет ответа"
                    else:
                        err_detail = str(exc)
                        if hasattr(exc, "code") and callable(getattr(exc, "code", None)):
                            try:
                                err_detail = f"{exc.code().name if hasattr(exc.code(), 'name') else exc.code()} — {err_detail}"
                            except Exception:
                                pass
                        logger.warning("InitShard failed for %s: %s", key, exc, exc_info=True)
                    errors.append(f"{key}: {err_detail}")
                    logger.warning(
                        "load_model_with_progress[%s]: InitShard finished with exception worker=%s shard=%s exc_detail=%s",
                        load_run_id,
                        key,
                        shard_id,
                        err_detail,
                    )
                workers_progress[key] = cluster_pb2.LoadProgress(stage="done", percent=100)
                progress_queue.put(make_event(make_master_progress("workers", 100), dict(workers_progress)))
                logger.info(
                    "load_model_with_progress[%s]: worker shard done worker=%s shard=%s",
                    load_run_id,
                    key,
                    shard_id,
                )

            if errors:
                logger.error("load_model_with_progress[%s]: FAILED errors=%s", load_run_id, errors)
                progress_queue.put(make_event(None, {}, done=True, ok=False, error="; ".join(errors)))
                return
            self._last_loaded_model_id = hf_model_id
            self._last_loaded_shard_ids = shard_id_list
            logger.info("load_model_with_progress[%s]: SUCCESS hf_model_id=%s shard_ids=%s", load_run_id, hf_model_id, shard_id_list)
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

    def get_cluster_logs(
        self,
        master_since: int,
        worker_since: Dict[str, int],
    ) -> cluster_pb2.GetClusterLogsResponse:
        """Собирает логи мастера и воркеров для отображения в GUI."""
        resp = cluster_pb2.GetClusterLogsResponse(master_next=master_since, worker_logs={})
        if self._log_buffer is not None:
            lines, next_idx = self._log_buffer.get_since(master_since)
            resp.master_lines.extend(lines)
            resp.master_next = next_idx
        with self._lock:
            stubs_snapshot = dict(self._stubs)
        for key, stub in stubs_snapshot.items():
            since = worker_since.get(key, 0)
            try:
                r = stub.GetWorkerLogs(
                    cluster_pb2.GetWorkerLogsRequest(since_index=since),
                    timeout=5.0,
                )
                chunk = cluster_pb2.WorkerLogsChunk(lines=list(r.lines), next_index=r.next_index)
                resp.worker_logs[key] = chunk
            except Exception:  # noqa: BLE001
                pass
        return resp

    def forward_chat_message_to_workers(
        self,
        header: cluster_pb2.ChatPostHeader,
        attachments: List[cluster_pb2.ChatPostAttachmentMeta],
    ) -> None:
        """
        Best-effort: переслать сообщение и вложения выбранным воркерам.
        Идемпотентность: один message_id пересылаем только один раз (защита от дублей при ретраях).
        """
        target_keys = list(header.target_worker_keys)
        if not target_keys:
            return
        mid = (header.message_id or "").strip()
        if not mid:
            return
        now = time.time()
        with self._lock:
            if mid in self._chat_forwarded_ids:
                return
            # Очистка старых записей
            expired = [k for k, t in self._chat_forwarded_ids.items() if now - t > self._chat_forwarded_ttl_s]
            for k in expired:
                del self._chat_forwarded_ids[k]
            while len(self._chat_forwarded_ids) >= self._chat_forwarded_max:
                oldest = min(self._chat_forwarded_ids.items(), key=lambda x: x[1])
                del self._chat_forwarded_ids[oldest[0]]
            self._chat_forwarded_ids[mid] = now

        def run() -> None:
            for worker_key in target_keys:
                stub = None
                with self._lock:
                    stub = self._stubs.get(worker_key)
                if stub is None:
                    continue

                def gen() -> Any:
                    # Заголовок (мастер отправляет и на воркер, тот же формат)
                    h = cluster_pb2.ChatPostHeader(
                        message_id=header.message_id,
                        timestamp_ms=header.timestamp_ms,
                        channel_id=header.channel_id,
                        sender=header.sender,
                        text=header.text,
                        target_worker_keys=[],  # на воркере не используется
                        attachments=list(attachments),
                    )
                    yield cluster_pb2.ChatPostChunk(header=h)

                    CHUNK = 256 * 1024
                    for a in attachments:
                        path = self._chat_storage.attachment_original_path(a.attachment_id)
                        try:
                            with path.open("rb") as f:
                                while True:
                                    b = f.read(CHUNK)
                                    if not b:
                                        break
                                    is_last = False
                                    # узнаём last через peek на размер следующего чанка
                                    # чтобы не читать лишнее — читаем один раз дальше.
                                    # (практически это доп. чтение на 1 чанку максимум)
                                    next_pos = f.tell()
                                    b2 = f.read(1)
                                    if b2 == b"":
                                        # на самом деле EOF
                                        is_last = True
                                        # вернуть обратно 0 байт не нужно
                                    else:
                                        # если есть следующий байт — переносим позицию назад
                                        f.seek(next_pos)
                                    yield cluster_pb2.ChatPostChunk(
                                        attachment_chunk=cluster_pb2.ChatPostAttachmentChunk(
                                            attachment_id=a.attachment_id,
                                            data=b,
                                            is_last=is_last,
                                        )
                                    )
                        except FileNotFoundError:
                            # Если файла нет — пропускаем, воркер получит ошибку по метаданным.
                            continue

                try:
                    # client-streaming -> unary response
                    stub.ReceiveChatMessage(gen(), timeout=300.0)
                except Exception:
                    # Best-effort: не ломаем master при проблемах пересылки.
                    continue

        threading.Thread(target=run, daemon=True).start()

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
