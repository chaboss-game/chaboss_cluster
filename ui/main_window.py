"""
PyQt6 UI для мониторинга кластера.
Вкладки: Воркеры, Настройки (адрес мастера, конфиг воркеров IP/port/key, модель HF, Скан, Старт).
Все настройки автосохраняются.
"""
from __future__ import annotations

import os
import sys
import subprocess
import threading
import uuid
import mimetypes
import base64
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import re

from PyQt6 import QtCore, QtGui, QtWidgets

from .tabs import ChatTabWidget, ClusterTabWidget, SettingsTabWidget, WorkerTabWidget

try:
    import grpc
    from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc
except ImportError as e:
    print(
        "Ошибка: сгенерируйте gRPC-модули: pip install -r requirements.txt && python scripts/gen_grpc.py",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from .settings_store import load as load_settings, save as save_settings

try:
    from cluster_core.common.config import load_master_config
except ImportError:
    load_master_config = None

DEFAULT_MASTER_ADDR = "127.0.0.1:60051"
POLL_INTERVAL_MS = 3000
LOG_POLL_MS = 2000  # опрос логов мастера и воркеров для отображения в окне логов
# Таймаут LoadModel: большие модели (HF download + рассылка воркерам) могут занимать 10–30+ минут
LOAD_MODEL_TIMEOUT_S = 3600
WORKER_VISIBILITY_POLL_MS = 5000  # опрос мастера в режиме воркера: виден ли воркер в реестре
AUTOSAVE_DELAY_MS = 500

MODEL_LOAD_MODES = [
    ("fit_in_cluster", "Модель влезает в кластер"),
    ("streaming_chunks", "Модель по частям (стриминг)"),
]

_STATUS_NAMES = {
    0: "ONLINE",
    1: "OFFLINE",
    2: "UNSTABLE",
    3: "RECONNECTING",
}


def _worker_list_to_dict(worker_list) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for w in worker_list.workers:
        key = f"{w.id.host}:{w.id.port}"
        out[key] = {
            "status": _STATUS_NAMES.get(w.status, "UNKNOWN"),
            "token_status": getattr(w, "token_status", "") or "",
            "os": ((getattr(w.resources, "os_name", "") or "") + " " + (getattr(w.resources, "os_version", "") or "")).strip(),
            "cpu_cores": w.resources.cpu_cores,
            "ram_total_mb": w.resources.ram_total_mb,
            "ram_available_mb": w.resources.ram_available_mb,
            "gpus": [{"name": g.name, "total_vram_mb": getattr(g, "total_vram_mb", 0)} for g in w.resources.gpus],
        }
    return out


class MasterPoller(QtCore.QObject):
    """
    Периодический опрос мастера (ListWorkers). Вызов poll() идёт по таймеру из MainWindow
    каждые POLL_INTERVAL_MS — чтобы обновлять таблицу воркеров. Попытки подключения к мастеру
    происходят не «при запуске мастера», а постоянно по таймеру; при ошибке в лог пишется
    сообщение через error.emit().
    """
    workers_updated = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, master_addr: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._master_addr = master_addr

    def set_master_addr(self, addr: str) -> None:
        self._master_addr = addr

    def poll(self) -> None:
        try:
            channel = grpc.insecure_channel(self._master_addr)
            stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
            response = stub.ListWorkers(cluster_pb2.Empty(), timeout=5.0)
            channel.close()
            self.workers_updated.emit(_worker_list_to_dict(response))
        except Exception as e:
            self.error.emit(str(e))


_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_host_port(addr: str) -> tuple[bool, str]:
    """
    Валидация адреса мастера формата host:port.
    Возвращает (ok, error_message).
    """
    a = (addr or "").strip()
    if not a or ":" not in a:
        return False, "Ожидается host:port (например, 127.0.0.1:60051)"
    host, port_s = a.rsplit(":", 1)
    host = host.strip()
    port_s = port_s.strip()
    if not host or not port_s:
        return False, "Ожидается host:port (например, 127.0.0.1:60051)"
    if not port_s.isdigit():
        return False, "Порт должен быть числом (1–65535)"
    port = int(port_s)
    if port < 1 or port > 65535:
        return False, "Порт должен быть в диапазоне 1–65535"
    if not _HOST_RE.match(host):
        return False, "Некорректный host (разрешены буквы/цифры/.-_)"
    return True, ""


class MainWindow(QtWidgets.QMainWindow):
    load_finished = QtCore.pyqtSignal(bool, str)
    load_progress_event = QtCore.pyqtSignal(object)  # cluster_pb2.LoadModelProgressEvent
    unload_finished = QtCore.pyqtSignal(bool, str)
    apply_workers_finished = QtCore.pyqtSignal(bool, str)
    log_message = QtCore.pyqtSignal(str)  # сообщение для вкладки «Лог» (с меткой времени добавляется в слоте)
    remote_log_lines = QtCore.pyqtSignal(list)  # строки логов мастера/воркеров без доп. метки времени
    log_since_updated = QtCore.pyqtSignal(int, dict)  # master_next, worker_since для следующего опроса
    worker_master_status = QtCore.pyqtSignal(str, str)  # (ключ воркера или "", статус в реестре мастера)

    # Чат
    chat_history_received = QtCore.pyqtSignal(object)  # list[dict]
    chat_channels_received = QtCore.pyqtSignal(list)  # list[(id,name)]
    chat_clipboard_set_text = QtCore.pyqtSignal(str)
    chat_clipboard_set_image = QtCore.pyqtSignal(object)
    chat_send_finished = QtCore.pyqtSignal(bool, str)
    chat_channels_mutation_finished = QtCore.pyqtSignal(bool, str)
    chat_workers_for_checklist_received = QtCore.pyqtSignal(dict)  # dict[str, dict]
    chat_receivers_refresh_failed = QtCore.pyqtSignal(str)  # ошибка опроса мастера для чеклиста (режим воркер)

    def __init__(self, master_addr: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Chaboss Cluster")
        self.load_finished.connect(self._on_load_finished)
        self.load_progress_event.connect(self._on_load_progress_event)
        self.unload_finished.connect(self._on_unload_finished)
        self.apply_workers_finished.connect(self._on_apply_workers_finished)
        self.log_message.connect(self._append_log_line)
        self.remote_log_lines.connect(self._append_remote_log_lines)
        self.log_since_updated.connect(self._on_log_since_updated)
        self.worker_master_status.connect(self._on_worker_master_status)

        # Чат
        self.chat_history_received.connect(self._chat_render_messages)
        self.chat_channels_received.connect(self._chat_render_channels)
        self.chat_clipboard_set_text.connect(self._chat_set_clipboard_text)
        self.chat_clipboard_set_image.connect(self._chat_set_clipboard_image)
        self.chat_send_finished.connect(self._chat_on_send_finished)
        self.chat_channels_mutation_finished.connect(self._chat_on_channels_mutation_finished)
        self.chat_workers_for_checklist_received.connect(self._on_chat_workers_for_checklist_received)
        self.chat_receivers_refresh_failed.connect(self._on_chat_receivers_refresh_failed)

        settings = load_settings()
        self._addr = (
            master_addr
            or os.environ.get("CLUSTER_MASTER_ADDR")
            or settings.get("master_addr", DEFAULT_MASTER_ADDR)
        ).strip()
        # Если адрес не передан явно, пытаемся взять актуальный из config/master.yaml.
        if master_addr is None and not os.environ.get("CLUSTER_MASTER_ADDR") and load_master_config is not None:
            try:
                project_root = Path(__file__).resolve().parent.parent
                cfg_path = project_root / "config" / "master.yaml"
                if cfg_path.exists():
                    cfg = load_master_config(cfg_path)
                    cfg_addr = f"{cfg.listen_host}:{cfg.listen_port}".strip()
                    if cfg_addr:
                        self._addr = cfg_addr
            except Exception:
                # Фолбэк на сохранённый адрес GUI.
                pass
        self._poller = MasterPoller(self._addr, self)
        self._last_workers_dict: Dict[str, dict] = {}
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._worker_visibility_timer = QtCore.QTimer(self)
        self._worker_visibility_timer.timeout.connect(self._poll_worker_visibility)
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_settings)
        self._logs_poll_timer = QtCore.QTimer(self)
        self._logs_poll_timer.timeout.connect(self._poll_cluster_logs)
        self._log_master_since = 0
        self._log_worker_since: Dict[str, int] = {}

        self._table_model = WorkerTableModel()
        workers_page = ClusterTabWidget(self, self._table_model, POLL_INTERVAL_MS).build()

        settings_page = SettingsTabWidget(self, MODEL_LOAD_MODES).build()

        chat_tab = ChatTabWidget(self).build()

        worker_tab = WorkerTabWidget(self).build()

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(workers_page, "Кластер")
        tabs.addTab(settings_page, "Настройки")
        tabs.addTab(chat_tab, "Чат")
        tabs.addTab(worker_tab, "Воркер")

        self.setCentralWidget(tabs)

        # Панель визуального прогресса загрузки модели (мастер + воркеры с живым %)
        self._load_stage_label = QtWidgets.QLabel("Ожидание запуска загрузки модели")
        self._load_progress_master = QtWidgets.QProgressBar()
        self._load_progress_master.setMinimum(0)
        self._load_progress_master.setMaximum(100)
        self._load_progress_master.setValue(0)
        self._load_progress_master.setFormat("Мастер: %p% (%v / %m)")
        self._load_workers_container = QtWidgets.QWidget()
        self._load_workers_layout = QtWidgets.QVBoxLayout(self._load_workers_container)
        self._load_workers_layout.setContentsMargins(0, 0, 0, 0)
        self._load_worker_bars: Dict[str, QtWidgets.QProgressBar] = {}
        load_prog_widget = QtWidgets.QWidget()
        lp_layout = QtWidgets.QVBoxLayout()
        lp_layout.setContentsMargins(6, 6, 6, 6)
        lp_layout.addWidget(QtWidgets.QLabel("Прогресс загрузки модели (HF → мастер → воркеры):"))
        lp_layout.addWidget(self._load_stage_label)
        lp_layout.addWidget(QtWidgets.QLabel("Мастер (скачивание с HF, разметка шардов):"))
        lp_layout.addWidget(self._load_progress_master)
        lp_layout.addWidget(QtWidgets.QLabel("Воркеры (скачивание с HF, загрузка шарда):"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._load_workers_container)
        scroll.setMaximumHeight(120)
        lp_layout.addWidget(scroll)
        load_prog_widget.setLayout(lp_layout)
        self._load_progress_dock = QtWidgets.QDockWidget("Прогресс модели", self)
        self._load_progress_dock.setWidget(load_prog_widget)
        self._load_progress_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._load_progress_dock)
        self._load_progress_dock.hide()
        load_prog_widget.setMinimumWidth(280)
        load_prog_widget.setMinimumHeight(120)
        self.statusBar().showMessage("Загрузка...")

        self._poller.workers_updated.connect(self._on_workers_updated)
        self._poller.error.connect(self._on_poller_error)
        # Таймер каждые POLL_INTERVAL_MS вызывает _on_timer → poll() → ListWorkers.
        # Так GUI постоянно обновляет список воркеров; попытки к мастеру идут по таймеру, не при старте мастера.
        self._timer.start(POLL_INTERVAL_MS)
        self._logs_poll_timer.start(LOG_POLL_MS)
        self._on_timer()

        # Применить загруженные настройки (модель, таблица воркеров, режим, ресурсы)
        self._model_edit.setText(settings.get("hf_model_id", "") or "")
        self._load_workers_into_table(settings.get("workers", []))
        mode_val = settings.get("model_load_mode", "fit_in_cluster")
        idx = self._mode_combo.findData(mode_val)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._resource_percent_spin.setValue(max(1, min(100, int(settings.get("resource_usage_percent", 75)))))
        self._worker_config_edit.setText(settings.get("worker_config_path", "config/worker.yaml") or "config/worker.yaml")

        # Обновлять адрес поллера при переключении на вкладку Воркеры или по таймеру
        self._apply_master_addr_from_edit()

        self._update_process_state()
        self.append_log("Запуск приложения. Подключение к мастеру: " + self._addr)

        # Чат: init и первая загрузка каналов
        self._chat_init_state()
        self._chat_load_channels()

    def _load_workers_into_table(self, workers: List[dict]) -> None:
        self._workers_table.blockSignals(True)
        self._workers_table.setRowCount(len(workers))
        for i, w in enumerate(workers):
            self._workers_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(w.get("host", ""))))
            self._workers_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(w.get("port", ""))))
            self._workers_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(w.get("auth_token", ""))))
        self._workers_table.blockSignals(False)

    def _get_workers_from_table(self) -> List[dict]:
        rows: List[dict] = []
        for i in range(self._workers_table.rowCount()):
            host = (self._workers_table.item(i, 0) or QtWidgets.QTableWidgetItem("")).text().strip()
            port_s = (self._workers_table.item(i, 1) or QtWidgets.QTableWidgetItem("")).text().strip()
            key = (self._workers_table.item(i, 2) or QtWidgets.QTableWidgetItem("")).text().strip()
            port = int(port_s) if port_s.isdigit() else 0
            rows.append({"host": host or "127.0.0.1", "port": port, "auth_token": key})
        return rows

    def _add_worker_row(self) -> None:
        r = self._workers_table.rowCount()
        self._workers_table.insertRow(r)
        self._workers_table.setItem(r, 0, QtWidgets.QTableWidgetItem("127.0.0.1"))
        self._workers_table.setItem(r, 1, QtWidgets.QTableWidgetItem("60052"))
        self._workers_table.setItem(r, 2, QtWidgets.QTableWidgetItem(""))
        self._schedule_save()

    def _remove_worker_row(self) -> None:
        row = self._workers_table.currentRow()
        if row >= 0:
            self._workers_table.removeRow(row)
            self._schedule_save()

    def _schedule_save(self) -> None:
        self._save_timer.stop()
        self._save_timer.start(AUTOSAVE_DELAY_MS)

    def _save_settings(self) -> None:
        addr = self._master_edit.text().strip() or DEFAULT_MASTER_ADDR
        if addr != self._addr:
            self._addr = addr
            self._poller.set_master_addr(addr)
            self._addr_label.setText(f"Мастер: {addr} (обновление каждые {POLL_INTERVAL_MS // 1000} с)")
        data = {
            "master_addr": addr,
            "hf_model_id": self._model_edit.text().strip(),
            "model_load_mode": self._mode_combo.currentData() or "fit_in_cluster",
            "resource_usage_percent": self._resource_percent_spin.value(),
            "workers": self._get_workers_from_table(),
            "worker_config_path": self._worker_config_edit.text().strip() or "config/worker.yaml",
        }
        save_settings(data)

    def append_log(self, message: str) -> None:
        """Добавить строку в окно логов (с меткой времени). Безопасно вызывать из любого потока."""
        if QtCore.QThread.currentThread() is self.thread():
            self._append_log_line(message)
        else:
            self.log_message.emit(message)

    def _append_log_line(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_text.appendPlainText(f"[{ts}] {message}")

    def _append_remote_log_lines(self, lines: List[str]) -> None:
        for line in lines:
            self._log_text.appendPlainText(line)

    def _on_log_since_updated(self, master_next: int, worker_since: dict) -> None:
        self._log_master_since = master_next
        self._log_worker_since = dict(worker_since) if worker_since else {}

    def _poll_cluster_logs(self) -> None:
        """Запросить логи мастера и воркеров и добавить новые строки в окно логов."""
        addr = getattr(self, "_addr", "") or ""
        if not addr:
            return
        master_since = getattr(self, "_log_master_since", 0)
        worker_since = getattr(self, "_log_worker_since", {})
        req = cluster_pb2.GetClusterLogsRequest(master_since=master_since, worker_since=worker_since)

        def do_fetch() -> None:
            try:
                channel = grpc.insecure_channel(addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                resp = stub.GetClusterLogs(req, timeout=5.0)
                channel.close()
            except Exception:
                return
            lines: List[str] = []
            for line in resp.master_lines:
                lines.append("[master] " + line)
            for key, chunk in resp.worker_logs.items():
                for line in chunk.lines:
                    lines.append(f"[{key}] " + line)
            if lines:
                self.remote_log_lines.emit(lines)
            worker_next = {k: v.next_index for k, v in resp.worker_logs.items()}
            self.log_since_updated.emit(resp.master_next, worker_next)
        threading.Thread(target=do_fetch, daemon=True).start()

    def _apply_master_addr_from_edit(self) -> None:
        addr = self._master_edit.text().strip() or DEFAULT_MASTER_ADDR
        ok, err = _validate_host_port(addr)
        if not ok:
            self.append_log("Некорректный адрес мастера: %s (%s)" % (addr, err))
            return
        if addr != self._addr:
            self._addr = addr
            self._poller.set_master_addr(addr)
            self._addr_label.setText(f"Мастер: {addr} (обновление каждые {POLL_INTERVAL_MS // 1000} с)")

    def _on_start_master(self) -> None:
        """Запуск мастера в отдельном процессе (python -m scripts.run_master)."""
        if self._master_process is not None and self._master_process.poll() is None:
            self.append_log("Мастер уже запущен (PID %s)" % self._master_process.pid)
            self.statusBar().showMessage(f"Мастер уже запущен (PID {self._master_process.pid})")
            return
        self.append_log("Запуск мастера (python -m scripts.run_master)...")
        project_root = Path(__file__).resolve().parent.parent
        try:
            env = os.environ.copy()
            env["CHABOSS_CLUSTER_FROM_GUI"] = "1"
            self._master_process = subprocess.Popen(
                [sys.executable, "-m", "scripts.run_master"],
                cwd=project_root,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.append_log("Мастер запущен, PID %s" % self._master_process.pid)
            self.statusBar().showMessage(f"Мастер запущен (PID {self._master_process.pid})")
            # Подставляем адрес мастера из config/master.yaml, чтобы GUI подключался к нему,
            # а не к старому порту воркера (если этот компьютер раньше был воркером).
            config_path = project_root / "config" / "master.yaml"
            if config_path.exists() and load_master_config is not None:
                try:
                    cfg = load_master_config(config_path)
                    addr = f"{cfg.listen_host}:{cfg.listen_port}"
                    self._addr = addr
                    self._master_edit.setText(addr)
                    self._poller.set_master_addr(addr)
                    self._addr_label.setText(f"Мастер: {addr} (обновление каждые {POLL_INTERVAL_MS // 1000} с)")
                    self._save_settings()
                    self.append_log("Адрес мастера установлен из конфига: %s" % addr)
                except Exception as e:  # noqa: BLE001
                    self.append_log("Не удалось прочитать адрес мастера из конфига: %s" % e)
            self._update_process_state()
            # Пауза опроса мастера на 2 с, чтобы мастер успел подняться — иначе в лог пойдут «connection refused».
            self._timer.stop()
            self._logs_poll_timer.stop()
            QtCore.QTimer.singleShot(2000, self._resume_master_poll)
        except Exception as e:
            self.append_log("Ошибка запуска мастера: %s" % e)
            self.statusBar().showMessage(f"Ошибка запуска мастера: {e}")
            self._master_process = None

    def _on_stop_master(self) -> None:
        """Остановка процесса мастера."""
        if self._master_process is None:
            self.append_log("Мастер не запущен")
            self.statusBar().showMessage("Мастер не запущен")
            return
        if self._master_process.poll() is not None:
            self.append_log("Мастер уже остановлен")
            self._master_process = None
            return
        try:
            self._master_process.terminate()
            self._master_process.wait(timeout=5)
        except Exception:
            try:
                self._master_process.kill()
            except Exception:
                pass
        self.append_log("Мастер остановлен (PID %s)" % getattr(self._master_process, "pid", ""))
        self._master_process = None
        self.statusBar().showMessage("Мастер остановлен")
        self._update_process_state()

    def _on_restart_master(self) -> None:
        """Перезапуск мастера: остановка, затем старт."""
        self._on_stop_master()
        if self._master_process is None:
            QtCore.QTimer.singleShot(500, self._on_start_master)

    def _refresh_worker_status_label(self) -> None:
        """Обновить текст панели воркера (процесс + статус в реестре мастера)."""
        port = self._worker_display_port if hasattr(self, "_worker_display_port") else None
        pid = getattr(self._worker_process, "pid", None) if getattr(self, "_worker_process", None) else None
        lines = [
            "Режим: Воркер",
            "Порт: %s" % (port or "—"),
            "Статус: работает (PID %s)" % (pid or "—") if pid is not None else "Статус: не запущен",
        ]
        if getattr(self, "_worker_master_status_text", ""):
            lines.append(self._worker_master_status_text)
        self._worker_status_label.setText("\n".join(lines))

    def _poll_worker_visibility(self) -> None:
        """В фоне опросить мастер (ListWorkers) и проверить, виден ли наш воркер по порту."""
        port = getattr(self, "_worker_display_port", None)
        addr = getattr(self, "_addr", DEFAULT_MASTER_ADDR)
        if not port:
            return
        ok, err = _validate_host_port(addr)
        if not ok:
            self.worker_master_status.emit("", f"ошибка адреса мастера: {err}")
            return

        def do_poll() -> None:
            try:
                channel = grpc.insecure_channel(addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                response = stub.ListWorkers(cluster_pb2.Empty(), timeout=5.0)
                channel.close()
                workers = _worker_list_to_dict(response)
                # В режиме "Воркер" обычный опрос мастера (MasterPoller) выключен,
                # а чеклист получателей в чате обновляется только через _chat_sync_workers_checklist.
                # Поэтому обновляем чеклист напрямую, но через сигнал (в UI-потоке).
                self.chat_workers_for_checklist_received.emit(workers)
                suffix = ":%s" % port
                for key, w in workers.items():
                    if key.endswith(suffix):
                        status = w.get("status", "UNKNOWN")
                        self.worker_master_status.emit(key, status)
                        return
                self.worker_master_status.emit("", "не найден")
            except Exception as e:
                err_short = e if len(str(e)) < 60 else str(e)[:57] + "..."
                self.worker_master_status.emit("", "ошибка: %s" % err_short)
                self.chat_receivers_refresh_failed.emit(str(e))

        threading.Thread(target=do_poll, daemon=True).start()

    def _on_worker_master_status(self, worker_key: str, status: str) -> None:
        """Обновить строку «В реестре мастера» и перерисовать панель воркера."""
        if worker_key:
            self._worker_master_status_text = "В реестре мастера: %s (%s)" % (status, worker_key)
        else:
            self._worker_master_status_text = "В реестре мастера: %s" % status
        if getattr(self, "_worker_process", None) is not None and self._worker_process.poll() is None:
            self._refresh_worker_status_label()

    def _on_chat_workers_for_checklist_received(self, workers: dict) -> None:
        """Обновить список получателей в чате (галочки ONLINE) в GUI-потоке."""
        try:
            self._chat_sync_workers_checklist(workers)
        except Exception:
            # чеклист может ещё не быть создан (на ранней стадии init)
            pass
        self._chat_on_receivers_poll_finished(workers)
        btn = getattr(self, "_chat_refresh_receivers_btn", None)
        if btn is not None:
            btn.setEnabled(True)

    def _on_chat_receivers_refresh_failed(self, err: str) -> None:
        """Опрос ListWorkers для чеклиста не удался (режим воркер)."""
        btn = getattr(self, "_chat_refresh_receivers_btn", None)
        if btn is not None:
            btn.setEnabled(True)
        if getattr(self, "_chat_receivers_refresh_pending", False):
            self._chat_receivers_refresh_pending = False
            self.append_log("Чат: обновление получателей не выполнено: %s" % (err or "неизвестная ошибка"))

    def _update_process_state(self) -> None:
        """Обновить вид кластера и блокировки: мастер и воркер взаимоисключающие."""
        master_running = self._master_process is not None and self._master_process.poll() is None
        worker_running = self._worker_process is not None and self._worker_process.poll() is None

        if worker_running:
            self._cluster_stack.setCurrentIndex(1)
            self._worker_master_status_text = "В реестре мастера: проверка..."
            self._refresh_worker_status_label()
            self._timer.stop()
            self._logs_poll_timer.stop()
            self._worker_visibility_timer.start(WORKER_VISIBILITY_POLL_MS)
            self._poll_worker_visibility()
            self._start_master_btn.setEnabled(False)
            self._stop_master_btn.setEnabled(False)
            self._restart_master_btn.setEnabled(False)
            self._master_grp.setEnabled(False)
            self._mode_grp.setEnabled(False)
            self._workers_grp.setEnabled(False)
            self._model_grp.setEnabled(False)
            self._start_worker_btn.setEnabled(False)
            self._stop_worker_btn.setEnabled(True)
            self._restart_worker_btn.setEnabled(True)
        elif master_running:
            self._cluster_stack.setCurrentIndex(0)
            self._worker_visibility_timer.stop()
            self._timer.start(POLL_INTERVAL_MS)
            self._logs_poll_timer.start(LOG_POLL_MS)
            self._start_master_btn.setEnabled(True)
            self._stop_master_btn.setEnabled(True)
            self._restart_master_btn.setEnabled(True)
            self._master_grp.setEnabled(True)
            self._mode_grp.setEnabled(True)
            self._workers_grp.setEnabled(True)
            self._model_grp.setEnabled(True)
            self._start_worker_btn.setEnabled(False)
            self._stop_worker_btn.setEnabled(False)
            self._restart_worker_btn.setEnabled(False)
        else:
            self._cluster_stack.setCurrentIndex(0)
            self._worker_master_status_text = ""
            self._worker_status_label.setText("Режим: Воркер\nПорт: —\nСтатус: не запущен")
            self._worker_visibility_timer.stop()
            self._timer.start(POLL_INTERVAL_MS)
            self._logs_poll_timer.start(LOG_POLL_MS)
            self._start_master_btn.setEnabled(True)
            self._stop_master_btn.setEnabled(True)
            self._restart_master_btn.setEnabled(True)
            self._master_grp.setEnabled(True)
            self._mode_grp.setEnabled(True)
            self._workers_grp.setEnabled(True)
            self._model_grp.setEnabled(True)
            self._start_worker_btn.setEnabled(True)
            self._stop_worker_btn.setEnabled(False)
            self._restart_worker_btn.setEnabled(False)

    def _on_start_worker(self) -> None:
        """Запуск воркера в отдельном процессе."""
        if self._worker_process is not None and self._worker_process.poll() is None:
            self.append_log("Воркер уже запущен (PID %s)" % self._worker_process.pid)
            self.statusBar().showMessage("Воркер уже запущен")
            return
        if self._master_process is not None and self._master_process.poll() is None:
            self.append_log("Невозможно запустить воркер: мастер уже запущен в этом окне")
            self.statusBar().showMessage("Сначала остановите мастера")
            return
        self._save_settings()
        project_root = Path(__file__).resolve().parent.parent
        config_path = self._worker_config_edit.text().strip() or "config/worker.yaml"
        path_resolved = (project_root / config_path).resolve()
        self._worker_display_port = None
        self._worker_display_host = None
        try:
            if path_resolved.exists():
                with open(path_resolved, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                self._worker_display_host = str(cfg.get("listen_host") or "").strip() or None
                self._worker_display_port = int(cfg.get("listen_port", 0)) or None
        except Exception:
            pass
        self.append_log("Запуск воркера (python -m scripts.run_worker --config %s)..." % config_path)
        try:
            self._worker_process = subprocess.Popen(
                [sys.executable, "-m", "scripts.run_worker", "--config", str(path_resolved)],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.append_log("Воркер запущен, PID %s" % self._worker_process.pid)
            self.statusBar().showMessage("Воркер запущен (PID %s)" % self._worker_process.pid)
        except Exception as e:
            self.append_log("Ошибка запуска воркера: %s" % e)
            self.statusBar().showMessage("Ошибка запуска воркера: %s" % e)
            self._worker_process = None
        self._update_process_state()

    def _on_stop_worker(self) -> None:
        """Остановка процесса воркера."""
        if self._worker_process is None:
            self.append_log("Воркер не запущен")
            return
        if self._worker_process.poll() is not None:
            self._worker_process = None
            self._update_process_state()
            return
        try:
            self._worker_process.terminate()
            self._worker_process.wait(timeout=5)
        except Exception:
            try:
                self._worker_process.kill()
            except Exception:
                pass
        self.append_log("Воркер остановлен (PID %s)" % getattr(self._worker_process, "pid", ""))
        self._worker_process = None
        self.statusBar().showMessage("Воркер остановлен")
        self._update_process_state()

    def _on_restart_worker(self) -> None:
        """Перезапуск воркера."""
        self._on_stop_worker()
        if self._worker_process is None:
            QtCore.QTimer.singleShot(500, self._on_start_worker)

    def _on_timer(self) -> None:
        """По таймеру (каждые POLL_INTERVAL_MS): опрос мастера ListWorkers для обновления таблицы воркеров."""
        threading.Thread(target=self._poller.poll, daemon=True).start()

    def _resume_master_poll(self) -> None:
        """Возобновить опрос мастера после паузы (например после запуска мастера)."""
        if getattr(self, "_worker_process", None) is None or self._worker_process.poll() is not None:
            self._timer.start(POLL_INTERVAL_MS)
            self._logs_poll_timer.start(LOG_POLL_MS)

    def _on_apply_workers_config(self) -> None:
        """Отправить конфиг воркеров на мастер (UpdateWorkersConfig)."""
        self._apply_master_addr_from_edit()
        self._save_settings()
        self.append_log("Применение конфига воркеров на мастере...")
        self.statusBar().showMessage("Применение конфига воркеров...")

        def do_apply():
            try:
                channel = grpc.insecure_channel(self._addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                workers = self._get_workers_from_table()
                req = cluster_pb2.UpdateWorkersConfigRequest(
                    workers=[
                        cluster_pb2.WorkerEndpoint(
                            host=w["host"],
                            port=w["port"],
                            auth_token=w.get("auth_token") or "",
                        )
                        for w in workers
                    ]
                )
                resp = stub.UpdateWorkersConfig(req, timeout=10.0)
                channel.close()
                self.apply_workers_finished.emit(resp.ok, resp.error or "")
            except Exception as e:
                self.apply_workers_finished.emit(False, str(e))

        threading.Thread(target=do_apply, daemon=True).start()

    def _on_apply_workers_finished(self, ok: bool, error: str) -> None:
        if ok:
            self.append_log("Конфиг воркеров успешно применён на мастере")
            self.statusBar().showMessage("Конфиг воркеров применён на мастере")
        else:
            self.append_log("Ошибка применения конфига воркеров: " + error)
            self.statusBar().showMessage(f"Ошибка применения конфига: {error}")

    def _on_remote_update_workers(self) -> None:
        """Попросить мастера обновить все воркеры: git pull + (опционально) рестарт GUI на воркерах."""
        self._apply_master_addr_from_edit()
        self._save_settings()
        self.append_log("Remote update воркеров: git pull + restart GUI (--start-worker)...")
        self.statusBar().showMessage("Remote update воркеров...")

        def do_update() -> None:
            try:
                channel = grpc.insecure_channel(self._addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                resp = stub.RemoteUpdateWorkers(
                    cluster_pb2.RemoteUpdateWorkersRequest(
                        restart_gui=True,
                        start_worker=True,
                        git_remote="origin",
                        git_branch="",
                    ),
                    timeout=300.0,
                )
                channel.close()
                if not resp.ok:
                    self.log_message.emit("Remote update: ошибка мастера: " + (resp.error or ""))
                for r in resp.results:
                    if r.ok:
                        msg = f"{r.worker}: ok"
                    else:
                        msg = f"{r.worker}: ошибка: {r.error}"
                        err_lower = (r.error or "").lower()
                        out_lower = (r.output or "").lower()
                        if "method not found" in err_lower or "unimplemented" in err_lower:
                            msg += "\n  → На этой машине воркер запущен со старой версией. Остановите воркер, сделайте git pull и перезапустите воркер вручную."
                        elif (
                            "unmerged" in err_lower
                            or "unmerged" in out_lower
                            or "conflict" in err_lower
                            or "conflict" in out_lower
                        ):
                            msg += (
                                "\n  → Локально на этом воркере незавершённое слияние/rebase. "
                                "Исправьте репозиторий вручную (см. вывод git выше), затем повторите Remote update."
                            )
                    if r.output:
                        out = r.output.strip().replace("\r\n", "\n")
                        if len(out) > 400:
                            out = out[:400] + "\n... (truncated)"
                        msg += "\n" + out
                    self.log_message.emit(msg)
                self.statusBar().showMessage("Remote update завершён")
            except Exception as e:
                self.log_message.emit("Remote update: исключение: " + str(e))
                self.statusBar().showMessage("Remote update: ошибка")

        threading.Thread(target=do_update, daemon=True).start()

    def _on_scan(self) -> None:
        self._apply_master_addr_from_edit()
        self._save_settings()
        self.append_log("Скан воркеров на мастере %s..." % self._addr)
        threading.Thread(target=self._poller.poll, daemon=True).start()
        self.statusBar().showMessage("Скан...")

    def _on_start(self) -> None:
        self._apply_master_addr_from_edit()
        self._save_settings()
        model_id = self._model_edit.text().strip()
        if not model_id:
            self.statusBar().showMessage("Введите название модели HuggingFace")
            return
        mode_val = self._mode_combo.currentData() or "fit_in_cluster"
        res_pct = self._resource_percent_spin.value()
        self.append_log(
            "Загрузка модели %s (режим: %s, ресурсы: %s%%)..."
            % (model_id, mode_val, res_pct)
        )
        self.statusBar().showMessage(f"Загрузка модели {model_id}...")

        # Визуальный прогресс: поток LoadModelStream даёт живой % мастера и воркеров
        self._load_progress_dock.show()
        self._load_progress_dock.raise_()
        self._load_stage_label.setText("Мастер: начало загрузки с HF...")
        self._load_progress_master.setValue(0)
        for bar in self._load_worker_bars.values():
            bar.setValue(0)
        # Отключаем повторные действия во время загрузки модели.
        # (Нужно, чтобы пользователь не случайно запустил повторный цикл загрузки.)
        model_grp_enabled = getattr(getattr(self, "_model_grp", None), "isEnabled", lambda: True)()
        if hasattr(self, "_start_model_btn") and self._start_model_btn is not None:
            self._start_model_btn.setEnabled(False)
        if hasattr(self, "_unload_model_btn") and self._unload_model_btn is not None:
            self._unload_model_btn.setEnabled(False)
        self._model_edit.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        def _ev_to_dict(ev) -> dict:
            """Перевод события в dict для безопасной передачи в GUI-поток (без protobuf)."""
            d = {
                "master_percent": ev.master.percent if ev.master else 0,
                "master_stage": (ev.master.stage or "download") if ev.master else "download",
                "master_current_file": (ev.master.current_file or "") if ev.master else "",
                "workers": {},
                "done": bool(ev.done),
                "ok": bool(ev.ok),
                "error": (ev.error or "") or "",
            }
            for k, p in ev.workers.items():
                d["workers"][k] = {"percent": p.percent if p else 0, "stage": (p.stage or "") if p else "", "current_file": (p.current_file or "") if p else ""}
            return d

        def do_load():
            self.log_message.emit("Подключение к мастеру %s для загрузки модели..." % self._addr)
            try:
                channel = grpc.insecure_channel(self._addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                self.log_message.emit("GUI: вызов LoadModelStream(model_id=%s)" % model_id)
                try:
                    stream = stub.LoadModelStream(
                        cluster_pb2.LoadModelRequest(hf_model_id=model_id),
                    )
                    first_event = True
                    for ev in stream:
                        if first_event:
                            stage = (ev.master.stage or "download") if ev.master else "download"
                            pct = int(ev.master.percent) if ev.master and ev.master.percent is not None else 0
                            cur_file = (ev.master.current_file or "") if ev.master else ""
                            self.log_message.emit(
                                f"GUI: первый progress-ev от мастера: stage={stage} percent={pct} file={cur_file}"
                            )
                            first_event = False
                            # Примечание: это лог на уровне GUI, без protobuf-разбора для protobuf-объекта.
                            # Стадию/проценты дальше покажет progress-bar, а подробные шаги появятся в remote-логах.
                            self.load_progress_event.emit(_ev_to_dict(ev))
                            if ev.done:
                                self.load_finished.emit(ev.ok, ev.error or "")
                                break
                            continue
                        self.load_progress_event.emit(_ev_to_dict(ev))
                        if ev.done:
                            self.log_message.emit("GUI: LoadModelStream done ok=%s error=%s" % (bool(ev.ok), (ev.error or "")[:200]))
                            self.load_finished.emit(ev.ok, ev.error or "")
                            break
                except grpc.RpcError as rpc_err:
                    details = rpc_err.details() or ""
                    code_name = rpc_err.code().name if hasattr(rpc_err, "code") and callable(getattr(rpc_err, "code", None)) else str(rpc_err.code())
                    self.log_message.emit(f"GUI: ошибка LoadModelStream RPC code={code_name} details={(details[:300] or '')}")
                    if rpc_err.code() == grpc.StatusCode.UNIMPLEMENTED or "Method not found" in (rpc_err.details() or ""):
                        self.log_message.emit("Мастер без LoadModelStream, используется загрузка без прогресса (LoadModel).")
                        self.load_progress_event.emit({
                            "master_percent": 0, "master_stage": "download", "master_current_file": "",
                            "workers": {}, "done": False, "ok": False, "error": "",
                        })
                        resp = stub.LoadModel(
                            cluster_pb2.LoadModelRequest(hf_model_id=model_id),
                            timeout=3600.0,
                        )
                        ev = cluster_pb2.LoadModelProgressEvent(
                            done=True, ok=resp.ok, error=resp.error or ""
                        )
                        self.load_progress_event.emit(_ev_to_dict(ev))
                        self.load_finished.emit(ev.ok, ev.error or "")
                    else:
                        raise
                channel.close()
            except Exception as e:
                err_msg = str(e)
                self.log_message.emit("Ошибка загрузки модели: " + err_msg)
                self.load_finished.emit(False, err_msg)

        # Сразу показываем 0%, чтобы док не был пустым до первого ответа мастера
        self.load_progress_event.emit({
            "master_percent": 0, "master_stage": "download", "master_current_file": "",
            "workers": {}, "done": False, "ok": False, "error": "",
        })
        threading.Thread(target=do_load, daemon=True).start()

    def _show_load_error_dialog(self, error: str) -> None:
        """Диалог при ошибке загрузки модели с подсказками по ресурсам и режиму."""
        deadline_hint = ""
        if "Deadline Exceeded" in error or "DEADLINE_EXCEEDED" in error:
            deadline_hint = (
                "• Ошибка «Deadline Exceeded»: сработал таймаут на стороне мастера или воркера. "
                "Для загрузки шардов (InitShard) дедлайн отключён; если ошибка повторяется — обновите мастер/воркеры до последней версии и проверьте логи.\n\n"
            )
        disk_hint = ""
        if "Недостаточно места" in error or "disk" in error.lower() or "os error 112" in error or "ENOSPC" in error:
            disk_hint = (
                "• Недостаточно места на диске (воркер): освободите место на том диске, где воркер хранит кэш HuggingFace "
                "(обычно ~/.cache/huggingface или %USERPROFILE%\\.cache\\huggingface на Windows). "
                "Либо задайте другую папку через переменную окружения HF_HOME / HUGGINGFACE_HUB_CACHE на воркере.\n\n"
            )
        swap_hint = ""
        if "os error 1455" in error or "Файл подкачки" in error or "pagefile" in error.lower():
            swap_hint = (
                "• На Windows не хватает файла подкачки (os error 1455). Увеличьте размер файла подкачки (pagefile) "
                "или освободите RAM. Для больших моделей при загрузке тензоров может требоваться заметный объём виртуальной памяти.\n\n"
            )
        reset_hint = ""
        if "Connection reset" in error or "connection reset" in error.lower() or "UNAVAILABLE" in error:
            reset_hint = (
                "• «Connection reset by peer» / UNAVAILABLE: соединение оборвалось со стороны воркера — "
                "чаще всего процесс воркера завершился (нехватка RAM/файла подкачки, падение Python). "
                "На машине этого воркера откройте logs/worker.log и журнал Windows (Просмотр событий). "
                "Попробуйте режим «Модель по частям (стриминг)» или увеличьте файл подкачки на этом узле.\n\n"
            )
        msg = (
            "Ошибка загрузки модели:\n\n%s\n\n"
            "Подробные причины смотрите в логах мастера и воркеров "
            "(консоль, где запущены master/worker, или журнал приложения).\n\n"
            "%s%s%s%s"
            "Рекомендации:\n"
            "• Если модель не влезает в выбранный %% свободных ресурсов — увеличьте "
            "«Использование свободных ресурсов (%%)» в настройках (например, до 80–90%%).\n"
            "• Если модель не влезает даже при 100%%:\n"
            "  (a) Переключитесь в режим «Модель по частям (стриминг)»;\n"
            "  (b) Добавьте воркеров в кластер для увеличения суммарных ресурсов."
        ) % (error, deadline_hint, disk_hint, swap_hint, reset_hint)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Ошибка загрузки модели")
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setText(msg)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        box.exec()

    def _on_load_progress_event(self, ev: dict) -> None:
        """Обновить прогресс-бары из потока LoadModelStream (ev — dict из _ev_to_dict)."""
        pct = ev.get("master_percent", 0)
        if pct >= 0:
            self._load_progress_master.setValue(min(100, pct))
        stage = ev.get("master_stage") or "download"
        cur_file = ev.get("master_current_file") or ""
        if cur_file:
            self._load_stage_label.setText("Мастер: %s — %s (%d%%)" % (stage, cur_file, pct))
        else:
            self._load_stage_label.setText("Мастер: %s (%d%%)" % (stage, pct))
        for key, prog in (ev.get("workers") or {}).items():
            if key not in self._load_worker_bars:
                bar = QtWidgets.QProgressBar()
                bar.setMinimum(0)
                bar.setMaximum(100)
                bar.setValue(0)
                bar.setFormat("%s: %%p%%" % key)
                self._load_worker_bars[key] = bar
                self._load_workers_layout.addWidget(bar)
            bar = self._load_worker_bars[key]
            bar.setValue(min(100, prog.get("percent", 0)))
            # Подсказка с этапом и текущим файлом
            stage = prog.get("stage") or "download"
            cur_file = prog.get("current_file") or ""
            if cur_file:
                bar.setToolTip("%s: %s — %s (%d%%)" % (key, stage, cur_file, prog.get("percent", 0)))
            else:
                bar.setToolTip("%s: %s (%d%%)" % (key, stage, prog.get("percent", 0)))
        if ev.get("done"):
            if ev.get("ok"):
                self._load_stage_label.setText("Модель успешно загружена: мастер и воркеры готовы")
            else:
                self._load_stage_label.setText("Ошибка при загрузке модели — см. сообщение ниже")

    def _on_load_finished(self, ok: bool, error: str) -> None:
        self._model_edit.setEnabled(True)
        # Возвращаем кнопки в исходное состояние после завершения загрузки.
        model_grp_enabled = getattr(getattr(self, "_model_grp", None), "isEnabled", lambda: True)()
        if hasattr(self, "_start_model_btn") and self._start_model_btn is not None:
            self._start_model_btn.setEnabled(bool(model_grp_enabled))
        if hasattr(self, "_unload_model_btn") and self._unload_model_btn is not None:
            self._unload_model_btn.setEnabled(bool(model_grp_enabled))
        if ok:
            self._load_progress_master.setValue(100)
            for bar in self._load_worker_bars.values():
                bar.setValue(100)
            self.append_log("Модель загружена и разослана воркерам")
            self.statusBar().showMessage("Модель загружена и разослана воркерам")
        else:
            self._load_progress_master.setValue(0)
            for bar in self._load_worker_bars.values():
                bar.setValue(0)
            self.append_log("Ошибка загрузки модели: " + error)
            self.statusBar().showMessage(f"Ошибка: {error}")
            self._show_load_error_dialog(error)

    def _on_unload_model(self) -> None:
        self.append_log("Выгрузка модели с воркеров...")
        self.statusBar().showMessage("Выгрузка модели с воркеров...")

        def do_unload() -> None:
            try:
                channel = grpc.insecure_channel(self._addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                resp = stub.UnloadModel(
                    cluster_pb2.UnloadModelRequest(model_id=""),
                    timeout=30.0,
                )
                channel.close()
                self.unload_finished.emit(resp.ok, resp.error or "")
            except Exception as e:
                self.unload_finished.emit(False, str(e))

        threading.Thread(target=do_unload, daemon=True).start()

    def _on_unload_finished(self, ok: bool, error: str) -> None:
        if ok:
            self.append_log("Модель выгружена с воркеров (VRAM освобождена)")
            self.statusBar().showMessage("Модель выгружена с воркеров (VRAM освобождена)")
        else:
            self.append_log("Ошибка выгрузки модели: " + error)
            self.statusBar().showMessage(f"Ошибка выгрузки: {error}")

    def _update_resources_label(self, workers: Dict[str, dict]) -> None:
        """Обновить подпись свободных ресурсов и примерного размера модели при выбранном %."""
        total_ram_mb = sum(w.get("ram_available_mb") or 0 for w in workers.values())
        total_vram_mb = 0
        for w in workers.values():
            for g in w.get("gpus", []):
                total_vram_mb += g.get("total_vram_mb") or 0
        pct = self._resource_percent_spin.value()
        budget_mb = int((total_ram_mb + total_vram_mb) * pct / 100.0)
        budget_gb = budget_mb / 1024.0
        if total_ram_mb + total_vram_mb > 0:
            self._resources_label.setText(
                "Свободно в кластере: RAM %d MB, VRAM %d MB. "
                "При выбранном %s%% под модель влезает примерно до %.2f GB."
                % (total_ram_mb, total_vram_mb, pct, budget_gb)
            )
        else:
            self._resources_label.setText("Свободные ресурсы: нет данных (выполните Скан)")

    def _on_workers_updated(self, workers: Dict[str, dict]) -> None:
        prev = getattr(self, "_last_workers_dict", {})
        self._table_model.update_workers(workers)
        n = len(workers)
        self.statusBar().showMessage(f"Воркеров: {n}")
        btn = getattr(self, "_chat_refresh_receivers_btn", None)
        if btn is not None:
            btn.setEnabled(True)

        # Лог: подключения, отключения, смена статуса
        for key in workers:
            if key not in prev:
                self.append_log("Воркер подключён: %s (статус: %s)" % (key, workers[key].get("status", "")))
        for key in prev:
            if key not in workers:
                self.append_log("Воркер отключён: %s" % key)
        for key in workers:
            if key in prev:
                old_s = prev[key].get("status", "")
                new_s = workers[key].get("status", "")
                if old_s != new_s:
                    self.append_log("Воркер %s: статус %s → %s" % (key, old_s, new_s))

        self._last_workers_dict = dict(workers)
        if hasattr(self, "_chat_workers_checklist"):
            self._chat_sync_workers_checklist(workers)
        self._chat_on_receivers_poll_finished(workers)
        self._update_resources_label(workers)

        # Блокировка работы с моделью при несоответствии токенов
        mismatched = [k for k, w in workers.items() if (w.get("token_status") or "") == "MISMATCH"]
        if mismatched:
            self.append_log("Token mismatch у воркеров: " + ", ".join(sorted(mismatched)) + " — работа с моделью запрещена")
            try:
                self._model_edit.setEnabled(False)
            except Exception:
                pass
        else:
            self._model_edit.setEnabled(True)

    def _on_poller_error(self, msg: str) -> None:
        self.append_log("Ошибка подключения к мастеру: " + msg)
        if getattr(self, "_chat_receivers_refresh_pending", False):
            self._chat_receivers_refresh_pending = False
            self.append_log("Чат: обновление получателей не выполнено (нет связи с мастером).")
        short = self._chat_format_error(msg)
        self.statusBar().showMessage(short)
        self._table_model.update_workers({})
        self._last_workers_dict = {}

    # =========================
    # Chat UI / logic
    # =========================

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        # Drag & Drop для вложений
        if hasattr(self, "_chat_drop_area") and obj is self._chat_drop_area:
            if event.type() == QtCore.QEvent.Type.DragEnter:
                mime = event.mimeData()
                if mime is not None and mime.hasUrls():
                    event.acceptProposedAction()
                    return True
            if event.type() == QtCore.QEvent.Type.Drop:
                mime = event.mimeData()
                paths: List[str] = []
                if mime is not None and mime.hasUrls():
                    for url in mime.urls():
                        if url.isLocalFile():
                            paths.append(url.toLocalFile())
                if paths:
                    self._chat_add_attachments(paths)
                event.acceptProposedAction()
                return True
        return super().eventFilter(obj, event)

    def _chat_init_state(self) -> None:
        if hasattr(self, "_chat_pending_attachments"):
            return
        self._chat_pending_attachments: List[dict] = []
        self._chat_messages: List[dict] = []
        self._chat_attachment_cache: Dict[str, bytes] = {}
        self._chat_last_seq = 0
        self._chat_active_channel_id: str = ""
        self._chat_send_in_progress = False
        self._chat_receivers_refresh_pending = False
        self._chat_channels_refresh_pending = False

        self._chat_poll_timer = QtCore.QTimer(self)
        self._chat_poll_timer.setInterval(2000)
        self._chat_poll_timer.timeout.connect(lambda: self._chat_poll_history(force=False))
        self._chat_poll_timer.start()

    def _chat_load_channels(self) -> None:
        """Подгрузить список каналов из мастера."""
        addr = getattr(self, "_addr", "") or ""
        if not addr:
            return

        def do_fetch() -> None:
            try:
                channel = grpc.insecure_channel(addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                resp = stub.ListChatChannels(cluster_pb2.Empty(), timeout=10.0)
                channel.close()
                channels = [(c.id or "", c.name or "") for c in resp.channels or [] if (c.id or "").strip()]
                self.chat_channels_received.emit(channels)
            except Exception as e:
                self.chat_channels_received.emit([])
                self.append_log("Ошибка загрузки каналов: %s" % e)

        threading.Thread(target=do_fetch, daemon=True).start()

    def _chat_render_channels(self, channels: List[tuple]) -> None:
        if not channels:
            # fallback: общий канал
            self._chat_channels_combo.clear()
            self._chat_channels_combo.addItem("general", userData="general")
            self._chat_active_channel_id = "general"
            self._chat_last_seq = 0
            if getattr(self, "_chat_channels_refresh_pending", False):
                self._chat_channels_refresh_pending = False
                self.append_log("Чат: каналы обновлены (всего: 1).")
            return

        current = self._chat_channels_combo.currentData() if hasattr(self, "_chat_channels_combo") else None
        self._chat_channels_combo.blockSignals(True)
        self._chat_channels_combo.clear()
        for cid, name in channels:
            cid = str(cid)
            name = str(name)
            if not cid:
                continue
            self._chat_channels_combo.addItem(name, userData=cid)
        self._chat_channels_combo.blockSignals(False)

        # выбор текущего или general
        idx = self._chat_channels_combo.findData("general")
        if idx < 0:
            idx = 0
        if idx >= 0:
            self._chat_channels_combo.setCurrentIndex(idx)
        else:
            self._chat_active_channel_id = "general"

        if getattr(self, "_chat_channels_refresh_pending", False):
            self._chat_channels_refresh_pending = False
            self.append_log("Чат: каналы обновлены (всего: %d)." % max(1, len(channels)))

    def _chat_name_to_id(self, name: str) -> str:
        base = (name or "").strip().lower()
        base = re.sub(r"[^a-z0-9_-]+", "_", base)
        base = base.strip("_")
        return base or f"channel_{uuid.uuid4().hex[:8]}"

    def _chat_create_channel(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Создать канал", "Имя канала:")
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            return
        channel_id = self._chat_name_to_id(name)

        self._chat_mutate_channels_async([
            {
                "type": cluster_pb2.ChatChannelMutationType.CHAT_CHANNEL_CREATE,
                "channel_id": channel_id,
                "name": name,
            }
        ])

    def _chat_rename_channel(self) -> None:
        if not getattr(self, "_chat_active_channel_id", ""):
            return
        current_id = str(self._chat_active_channel_id)
        current_name = ""
        # вытащим текущий name из combo (если есть)
        for i in range(self._chat_channels_combo.count()):
            if self._chat_channels_combo.itemData(i) == current_id:
                current_name = self._chat_channels_combo.itemText(i)
                break
        name, ok = QtWidgets.QInputDialog.getText(self, "Переименовать канал", "Новое имя канала:", text=current_name)
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            return
        self._chat_mutate_channels_async([
            {
                "type": cluster_pb2.ChatChannelMutationType.CHAT_CHANNEL_RENAME,
                "channel_id": current_id,
                "name": name,
            }
        ])

    def _chat_delete_channel(self) -> None:
        if not getattr(self, "_chat_active_channel_id", ""):
            return
        current_id = str(self._chat_active_channel_id)
        if current_id == "general":
            QtWidgets.QMessageBox.warning(self, "Чат", "Канал general нельзя удалить.")
            return
        resp = QtWidgets.QMessageBox.question(self, "Удалить канал", f"Удалить канал '{current_id}'?")
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._chat_mutate_channels_async([
            {
                "type": cluster_pb2.ChatChannelMutationType.CHAT_CHANNEL_DELETE,
                "channel_id": current_id,
                "name": "",
            }
        ])

    def _chat_mutate_channels_async(self, ops: List[dict]) -> None:
        addr = getattr(self, "_addr", "") or ""
        if not addr:
            QtWidgets.QMessageBox.warning(self, "Чат", "Мастер не задан/недоступен")
            return

        def do_mutate() -> None:
            try:
                ch = grpc.insecure_channel(addr)
                try:
                    stub = cluster_pb2_grpc.MasterAdminServiceStub(ch)
                    req = cluster_pb2.ChatChannelsMutationRequest(
                        ops=[
                            cluster_pb2.ChatChannelMutation(
                                type=o["type"],
                                channel_id=o["channel_id"],
                                name=o.get("name") or "",
                            )
                            for o in ops
                        ]
                    )
                    # ChatChannelsResponse содержит только список `channels`,
                    # поэтому признак успеха определяется фактом отсутствия RPC-ошибки.
                    stub.MutateChatChannels(req, timeout=10.0)
                finally:
                    ch.close()
            except Exception as e:
                self.chat_channels_mutation_finished.emit(False, str(e))
                return
            self.chat_channels_mutation_finished.emit(True, "")

        threading.Thread(target=do_mutate, daemon=True).start()

    def _chat_on_channels_mutation_finished(self, ok: bool, error: str) -> None:
        if ok:
            self._chat_load_channels()
            QtWidgets.QMessageBox.information(self, "Чат", "Каналы обновлены")
            return
        err = self._chat_format_error(error)
        self.append_log("Ошибка изменения каналов: %s" % err)
        QtWidgets.QMessageBox.warning(self, "Чат", err)

    def _chat_on_channel_changed(self) -> None:
        cid = self._chat_channels_combo.currentData()
        cid = str(cid or "").strip()
        if not cid:
            return
        self._chat_active_channel_id = cid
        self._chat_last_seq = 0
        self._chat_messages = []
        self._chat_messages_list.clear()
        self._chat_clear_attachments_panel()
        self._chat_poll_history(force=True)

    def _chat_clear_attachments_panel(self) -> None:
        while self._chat_attachments_layout.count():
            item = self._chat_attachments_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.deleteLater()

    def _chat_sync_workers_checklist(self, workers: Dict[str, dict]) -> None:
        if not hasattr(self, "_chat_workers_checklist"):
            return
        original_worker_keys = set(workers.keys())
        self_host = getattr(self, "_worker_display_host", None)
        self_port = getattr(self, "_worker_display_port", None)
        self_key = None
        if self_host and self_port:
            self_key = f"{self_host}:{self_port}"

        # В режиме "Воркер" хотим:
        # 1) не показывать получателем сам себя;
        # 2) показать мастер как опцию (сообщение всё равно сохраняется в мастере).
        worker_mode = getattr(self, "_worker_process", None) is not None and self._worker_process.poll() is None
        master_key = getattr(self, "_addr", "") or ""
        if worker_mode and master_key:
            # Безопасная логика для чеклиста в режиме "воркер":
            # - не удаляем ничего из реестра (чтобы не получить пустой чеклист из-за совпадений портов),
            # - гарантируем запись мастера под точным ключом `_addr`,
            # - мастер помечаем ONLINE.
            workers = dict(workers)
            w_master = dict(workers.get(master_key) or {})
            w_master["status"] = "ONLINE"
            w_master.setdefault("token_status", "")
            w_master.setdefault("os", "")
            w_master.setdefault("cpu_cores", 0)
            w_master.setdefault("ram_total_mb", 0)
            w_master.setdefault("ram_available_mb", 0)
            w_master.setdefault("gpus", [])
            workers[master_key] = w_master
        # Сохраняем выбранные ключи, чтобы не сбрасывать галочки при обновлении списка
        checked_keys: set = set()
        for i in range(self._chat_workers_checklist.count()):
            item = self._chat_workers_checklist.item(i)
            if item and item.checkState() == QtCore.Qt.CheckState.Checked:
                k = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if k:
                    checked_keys.add(str(k))
        self._chat_workers_checklist.blockSignals(True)
        self._chat_workers_checklist.clear()
        for key, w in sorted(workers.items()):
            # Пропускаем "себя" только если этот endpoint реально пришёл от мастера.
            # Иначе можем случайно пропустить мастер-опцию, которую мы добавляем вручную ниже.
            if self_key and str(key).strip() == self_key and str(key).strip() in original_worker_keys:
                continue
            status_val = w.get("status")
            if str(status_val or "").strip().upper() != "ONLINE":
                continue
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if key in checked_keys else QtCore.Qt.CheckState.Unchecked
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            self._chat_workers_checklist.addItem(item)
        self._chat_workers_checklist.blockSignals(False)

    def _chat_on_receivers_poll_finished(self, workers: Dict[str, dict]) -> None:
        """Один раз логирует результат после ручного обновления чеклиста получателей."""
        if not getattr(self, "_chat_receivers_refresh_pending", False):
            return
        self._chat_receivers_refresh_pending = False
        online_n = sum(1 for w in workers.values() if (w.get("status") or "").upper() == "ONLINE")
        self.append_log(
            "Чат: получатели обновлены (в реестре мастера: %d, ONLINE: %d)."
            % (len(workers), online_n)
        )

    def _chat_refresh_receivers(self) -> None:
        """
        Принудительно обновить список получателей (чеклист ONLINE).

        В режиме "Воркер" делаем один опрос через `_poll_worker_visibility`,
        в режиме "Мастер" — запускаем `MasterPoller.poll()` в фоне.
        """
        btn = getattr(self, "_chat_refresh_receivers_btn", None)
        if btn is not None:
            btn.setEnabled(False)

        worker_process = getattr(self, "_worker_process", None)
        if worker_process is not None and worker_process.poll() is None:
            port = getattr(self, "_worker_display_port", None)
            addr = getattr(self, "_addr", DEFAULT_MASTER_ADDR)
            if not port:
                self.append_log(
                    "Чат: обновление получателей отменено — не задан порт воркера (см. config/worker.yaml)."
                )
                if btn is not None:
                    btn.setEnabled(True)
                return
            ok_addr, addr_err = _validate_host_port(addr)
            if not ok_addr:
                self.append_log(
                    "Чат: обновление получателей отменено — некорректный адрес мастера: %s" % addr_err
                )
                if btn is not None:
                    btn.setEnabled(True)
                return
            self._chat_receivers_refresh_pending = True
            self.append_log(
                "Чат: запрошено обновление получателей (режим воркер, опрос мастера ListWorkers)…"
            )
            self._poll_worker_visibility()
            return

        self._chat_receivers_refresh_pending = True
        self.append_log(
            "Чат: запрошено обновление получателей (режим мастер, опрос ListWorkers)…"
        )
        threading.Thread(target=self._poller.poll, daemon=True).start()

    def _chat_refresh_channels_and_receivers(self) -> None:
        """
        Обновить сразу список каналов (подгрузка с мастера) и получателей онлайн.
        Кнопка привязана к этому методу.
        """
        btn = getattr(self, "_chat_refresh_receivers_btn", None)
        if btn is not None:
            btn.setEnabled(False)

        self._chat_channels_refresh_pending = True
        self.append_log("Чат: запрошено обновление каналов и получателей…")

        addr = getattr(self, "_addr", "") or ""
        if not addr:
            self._chat_channels_refresh_pending = False
            self.append_log("Чат: обновление каналов отменено — мастер не задан.")
        else:
            self._chat_load_channels()

        # Обновляем получателей (этот метод также управляет повторным включением кнопки)
        self._chat_refresh_receivers()

    def _chat_add_attachments(self, paths: List[str]) -> None:
        if not paths:
            return
        if len(self._chat_pending_attachments) >= 5:
            self.append_log("Нельзя добавить больше 5 вложений в одно сообщение")
            return
        for p in paths:
            if len(self._chat_pending_attachments) >= 5:
                break
            try:
                file_path = str(p)
                if not os.path.isfile(file_path):
                    continue
                size = os.path.getsize(file_path)
                if size > 20 * 1024 * 1024:
                    self.append_log("Файл слишком большой (20MB max): %s" % os.path.basename(file_path))
                    continue
                filename = os.path.basename(file_path)
                mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                is_image = mime_type.lower().startswith("image/")

                thumb_bytes = b""
                if is_image:
                    qimg = QtGui.QImage(file_path)
                    if not qimg.isNull():
                        qimg2 = qimg.scaled(
                            256,
                            256,
                            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation,
                        )
                        buf = QtCore.QBuffer()
                        buf.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)
                        qimg2.save(buf, "JPEG")
                        thumb_bytes = bytes(buf.data())
                        buf.close()
                    else:
                        is_image = False
                        mime_type = "application/octet-stream"

                att = {
                    "attachment_id": uuid.uuid4().hex,
                    "path": file_path,
                    "filename": filename,
                    "mime_type": mime_type,
                    "is_image": bool(is_image),
                    "size": int(size),
                    "thumbnail_jpeg": thumb_bytes,
                }
                self._chat_pending_attachments.append(att)
            except Exception as e:
                self.append_log("Ошибка добавления файла: %s" % e)

        self._chat_refresh_selected_files_list()

    def _chat_remove_pending_attachment(self, attachment_id: str) -> None:
        aid = (attachment_id or "").strip()
        if not aid:
            return
        self._chat_pending_attachments = [a for a in self._chat_pending_attachments if (a.get("attachment_id") or "") != aid]
        self._chat_refresh_selected_files_list()

    def _chat_refresh_selected_files_list(self) -> None:
        self._chat_selected_files_list.clear()
        for att in self._chat_pending_attachments:
            filename = str(att.get("filename") or att.get("path") or "file")
            size = int(att.get("size") or 0)
            size_kb = max(1, size // 1024) if size > 0 else 0
            aid = str(att.get("attachment_id") or "")

            item = QtWidgets.QListWidgetItem()
            row = QtWidgets.QWidget()
            row_l = QtWidgets.QHBoxLayout()
            row_l.setContentsMargins(6, 2, 6, 2)
            row_l.setSpacing(6)

            label = QtWidgets.QLabel(f"{filename} ({size_kb} KB)")
            label.setToolTip(att.get("path") or filename)
            label.setWordWrap(False)
            row_l.addWidget(label, 1)

            remove_btn = QtWidgets.QToolButton()
            remove_btn.setText("×")
            remove_btn.setToolTip("Убрать файл из отправки")
            remove_btn.clicked.connect(lambda _=False, x=aid: self._chat_remove_pending_attachment(x))
            row_l.addWidget(remove_btn, 0)

            row.setLayout(row_l)
            item.setSizeHint(row.sizeHint())
            self._chat_selected_files_list.addItem(item)
            self._chat_selected_files_list.setItemWidget(item, row)

        has_files = len(self._chat_pending_attachments) > 0
        self._chat_selected_files_label.setVisible(has_files)
        self._chat_selected_files_list.setVisible(has_files)

    def _chat_pick_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Выберите файлы для чата",
            "",
            "All files (*)",
        )
        if paths:
            self._chat_add_attachments(paths)

    def _chat_poll_history(self, force: bool) -> None:
        if not getattr(self, "_chat_active_channel_id", ""):
            return
        addr = getattr(self, "_addr", "") or ""
        if not addr:
            return

        since_seq = 0 if force else int(getattr(self, "_chat_last_seq", 0) or 0)
        channel_id = self._chat_active_channel_id

        def do_fetch() -> None:
            try:
                ch = grpc.insecure_channel(addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(ch)
                resp = stub.GetChatHistory(
                    cluster_pb2.GetChatHistoryRequest(
                        channel_id=channel_id,
                        since_seq=since_seq,
                        limit=200,
                    ),
                    timeout=10.0,
                )
                ch.close()
                msgs: List[dict] = []
                for m in resp.messages or []:
                    atts: List[dict] = []
                    for a in m.attachments or []:
                        atts.append(
                            {
                                "attachment_id": a.attachment_id,
                                "filename": a.filename,
                                "mime_type": a.mime_type,
                                "is_image": bool(a.is_image),
                                "size": int(a.size or 0),
                                "thumbnail_jpeg": bytes(a.thumbnail_jpeg or b""),
                            }
                        )
                    msgs.append(
                        {
                            "message_id": m.message_id,
                            "seq": int(m.seq or 0),
                            "timestamp_ms": int(m.timestamp_ms or 0),
                            "channel_id": m.channel_id,
                            "sender": m.sender or "",
                            "text": m.text or "",
                            "attachments": atts,
                        }
                    )
                payload = {"messages": msgs, "next_seq": int(resp.next_seq or 0)}
                self.chat_history_received.emit(payload)
            except Exception as e:
                # без диалога — чат должен жить даже при временных сетевых проблемах
                return

        threading.Thread(target=do_fetch, daemon=True).start()

    def _chat_render_messages(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        msgs = payload.get("messages") or []
        next_seq = int(payload.get("next_seq") or 0)
        if not msgs and next_seq == getattr(self, "_chat_last_seq", 0):
            return

        for m in msgs:
            text = (m.get("text") or "").strip()
            sender = m.get("sender") or ""
            ts_ms = int(m.get("timestamp_ms") or 0)
            try:
                dt = datetime.fromtimestamp(ts_ms / 1000.0) if ts_ms else datetime.now()
                time_str = dt.strftime("%d.%m.%Y %H:%M")
            except Exception:
                time_str = ""
            msg_text = text if text else "(без текста)"
            # Полный текст должен отображаться с переносами строк (без горизонтального скролла).
            item = QtWidgets.QListWidgetItem(f"[{time_str}] {sender}:\n{msg_text}")
            # иконка для первого изображения
            icon_pixmap = None
            for a in m.get("attachments") or []:
                if a.get("is_image") and a.get("thumbnail_jpeg"):
                    pm = QtGui.QPixmap()
                    pm.loadFromData(a["thumbnail_jpeg"], "JPEG")
                    icon_pixmap = pm
                    break
            if icon_pixmap is not None and not icon_pixmap.isNull():
                item.setIcon(QtGui.QIcon(icon_pixmap))
            self._chat_messages.append(m)
            self._chat_messages_list.addItem(item)

        self._chat_last_seq = next_seq
        # если ещё нет выделения — выделим первый
        if self._chat_messages_list.currentRow() < 0 and self._chat_messages:
            self._chat_messages_list.setCurrentRow(0)

    def _chat_on_message_selected(self, row: int) -> None:
        self._chat_clear_attachments_panel()
        if row < 0 or row >= len(self._chat_messages):
            return
        m = self._chat_messages[row]
        mid = m.get("message_id") or ""
        for a in m.get("attachments") or []:
            att_id = a.get("attachment_id") or ""
            mime_type = a.get("mime_type") or "application/octet-stream"
            filename = a.get("filename") or att_id
            is_image = bool(a.get("is_image") or False)
            thumb = a.get("thumbnail_jpeg") or b""

            row_w = QtWidgets.QWidget()
            row_l = QtWidgets.QHBoxLayout()
            row_l.setContentsMargins(0, 0, 0, 0)
            label_txt = f"{filename}"
            label = QtWidgets.QLabel(label_txt)
            label.setWordWrap(True)
            row_l.addWidget(label, 1)
            if is_image and thumb:
                pm = QtGui.QPixmap()
                pm.loadFromData(thumb, "JPEG")
                pm = pm.scaled(80, 80, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                img_lbl = QtWidgets.QLabel()
                img_lbl.setPixmap(pm)
                row_l.addWidget(img_lbl)
            copy_btn = QtWidgets.QPushButton("Копировать")
            copy_btn.clicked.connect(lambda _=False, mid=mid, aid=att_id, is_img=is_image, mt=mime_type: self._chat_copy_attachment(mid, aid, is_img, mt))
            row_l.addWidget(copy_btn)
            row_w.setLayout(row_l)
            self._chat_attachments_layout.addWidget(row_w)

    def _chat_copy_attachment(self, message_id: str, attachment_id: str, is_image: bool, mime_type: str) -> None:
        if not attachment_id:
            return
        if attachment_id in self._chat_attachment_cache:
            data = self._chat_attachment_cache[attachment_id]
            self._chat_apply_clipboard_data(attachment_id, data, is_image, mime_type)
            return

        addr = getattr(self, "_addr", "") or ""
        if not addr:
            return

        self._chat_send_status.setText("Копирование...")

        def do_copy() -> None:
            try:
                ch = grpc.insecure_channel(addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(ch)
                stream = stub.GetChatAttachment(
                    cluster_pb2.GetChatAttachmentRequest(message_id=message_id, attachment_id=attachment_id),
                    timeout=600.0,
                )
                buf = bytearray()
                for c in stream:
                    if c.data:
                        buf.extend(c.data)
                    if c.is_last:
                        break
                ch.close()
                data = bytes(buf)
                self._chat_attachment_cache[attachment_id] = data
                self._chat_apply_clipboard_from_thread(data, is_image, mime_type)
            except Exception as e:  # noqa: BLE001
                self.append_log("Ошибка копирования: %s" % e)

        threading.Thread(target=do_copy, daemon=True).start()

    def _chat_apply_clipboard_from_thread(self, data: bytes, is_image: bool, mime_type: str) -> None:
        if is_image:
            img = QtGui.QImage.fromData(data)
            if img.isNull():
                self.append_log("Не удалось распознать изображение для clipboard")
                return
            self.chat_clipboard_set_image.emit(img)
        else:
            b64 = base64.b64encode(data).decode("ascii")
            prefix = mime_type or "application/octet-stream"
            self.chat_clipboard_set_text.emit(f"data:{prefix};base64,{b64}")

    def _chat_apply_clipboard_data(self, attachment_id: str, data: bytes, is_image: bool, mime_type: str) -> None:
        # используется только из кэша (в GUI потоке)
        self._chat_apply_clipboard_from_thread(data, is_image, mime_type)

    def _chat_set_clipboard_text(self, text: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text)
        self._chat_send_status.setText("Готово")

    def _chat_set_clipboard_image(self, img: object) -> None:
        # img как QtGui.QImage
        try:
            QtWidgets.QApplication.clipboard().setImage(img)  # type: ignore[arg-type]
        except Exception:
            pass
        self._chat_send_status.setText("Готово")

    def _chat_on_send_finished(self, ok: bool, error: str) -> None:
        # Вызывается в GUI-потоке по сигналу — все обновления UI только здесь.
        self._chat_send_in_progress = False
        self._chat_send_btn.setEnabled(True)
        if ok:
            self._chat_text_edit.setPlainText("")
            self._chat_pending_attachments = []
            self._chat_refresh_selected_files_list()
            self._chat_send_status.setStyleSheet("color: #1f7a1f;")
            self._chat_send_status.setText("Отправлено")
            self._chat_poll_history(force=False)
            self.append_log("Чат: сообщение отправлено")
            if error:
                self.append_log(error)
        else:
            err = self._chat_format_error(error)
            self._chat_send_status.setStyleSheet("color: #b42318;")
            self._chat_send_status.setText(err)
            self.append_log("Чат: ошибка отправки: " + (error or ""))
            QtWidgets.QMessageBox.warning(self, "Чат", err)

    def _chat_send_message(self) -> None:
        if getattr(self, "_chat_send_in_progress", False):
            return
        if not getattr(self, "_chat_pending_attachments", None):
            self._chat_pending_attachments = []

        channel_id = str(self._chat_channels_combo.currentData() or "").strip()
        if not channel_id:
            QtWidgets.QMessageBox.warning(self, "Чат", "Выберите канал")
            return

        target_keys: List[str] = []
        for i in range(self._chat_workers_checklist.count()):
            item = self._chat_workers_checklist.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                key = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if key:
                    target_keys.append(str(key))
        if not target_keys:
            QtWidgets.QMessageBox.warning(self, "Чат", "Выберите хотя бы одного получателя (воркер ONLINE).")
            return

        text = (self._chat_text_edit.toPlainText() or "").strip()

        # сбор заголовка и метаданных вложений
        attachments_meta: List[cluster_pb2.ChatPostAttachmentMeta] = []
        for att in self._chat_pending_attachments[:5]:
            thumb = att.get("thumbnail_jpeg") or b""
            attachments_meta.append(
                cluster_pb2.ChatPostAttachmentMeta(
                    attachment_id=att["attachment_id"],
                    filename=att.get("filename") or "file",
                    mime_type=att.get("mime_type") or "application/octet-stream",
                    is_image=bool(att.get("is_image") or False),
                    size=int(att.get("size") or 0),
                    thumbnail_jpeg=thumb if bool(att.get("is_image") or False) else b"",
                )
            )

        message_id = uuid.uuid4().hex
        timestamp_ms = int(time.time() * 1000)
        sender = os.environ.get("COMPUTERNAME") or "gui"

        header = cluster_pb2.ChatPostHeader(
            message_id=message_id,
            timestamp_ms=timestamp_ms,
            channel_id=channel_id,
            sender=sender,
            text=text,
            target_worker_keys=target_keys,
            attachments=attachments_meta,
        )

        addr = getattr(self, "_addr", "") or ""
        if not addr:
            QtWidgets.QMessageBox.warning(self, "Чат", "Мастер не задан/недоступен")
            return

        self._chat_send_btn.setEnabled(False)
        self._chat_send_status.setStyleSheet("")
        self._chat_send_status.setText("Отправка...")
        self.append_log(f"Чат: отправка сообщения {message_id} (канал={channel_id}, вложений={len(self._chat_pending_attachments)})")

        def gen() -> Any:
            yield cluster_pb2.ChatPostChunk(header=header)
            CHUNK = 256 * 1024
            for att in self._chat_pending_attachments[:5]:
                path = att.get("path")
                aid = att.get("attachment_id")
                if not path or not aid:
                    continue
                with open(path, "rb") as f:
                    b = f.read(CHUNK)
                    if not b:
                        # Нулевой размер: всё равно нужно отправить один чанкуемый блок.
                        yield cluster_pb2.ChatPostChunk(
                            attachment_chunk=cluster_pb2.ChatPostAttachmentChunk(
                                attachment_id=aid,
                                data=b"",
                                is_last=True,
                            )
                        )
                        continue
                    while b:
                        b2 = f.read(CHUNK)
                        is_last = b2 == b""
                        yield cluster_pb2.ChatPostChunk(
                            attachment_chunk=cluster_pb2.ChatPostAttachmentChunk(
                                attachment_id=aid,
                                data=b,
                                is_last=is_last,
                            )
                        )
                        b = b2

        def do_send() -> None:
            # Не трогаем UI из потока — только gRPC и emit в GUI-поток.
            try:
                ch = grpc.insecure_channel(addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(ch)
                resp = stub.PostChatMessage(gen(), timeout=600.0)
                ch.close()
                if resp.ok:
                    self.chat_send_finished.emit(True, "")
                else:
                    self.chat_send_finished.emit(False, resp.error or "Ошибка отправки")
            except Exception as e:  # noqa: BLE001
                self.chat_send_finished.emit(False, str(e))

        self._chat_send_in_progress = True
        threading.Thread(target=do_send, daemon=True).start()

    # end chat

    def _chat_format_error(self, raw_error: str) -> str:
        msg = (raw_error or "").strip()
        if not msg:
            return "Неизвестная ошибка чата."
        low = msg.lower()
        if "statuscode.unavailable" in low or "failed to connect to all addresses" in low:
            return (
                "Нет соединения с мастером. Проверьте адрес мастера в настройках и убедитесь, "
                "что мастер запущен."
            )
        if "deadline_exceeded" in low or "deadline exceeded" in low:
            return "Таймаут запроса к мастеру. Проверьте нагрузку и сетевую доступность."
        if "statuscode.unimplemented" in low or "method not found" in low:
            return "Несовместимые версии GUI и мастера. Обновите проект на всех узлах."
        if len(msg) > 220:
            return msg[:217] + "..."
        return msg


class WorkerTableModel(QtCore.QAbstractTableModel):
    HEADERS = ["ID", "Status", "Token", "OS", "CPU", "RAM свободно / всего (MB)", "GPUs"]

    def __init__(self, workers: Dict[str, dict] | None = None) -> None:
        super().__init__()
        self._workers: Dict[str, dict] = workers or {}
        self._keys = list(self._workers.keys())

    def update_workers(self, workers: Dict[str, dict]) -> None:
        self.beginResetModel()
        self._workers = workers
        self._keys = list(self._workers.keys())
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self._keys)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self.HEADERS)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        worker_key = self._keys[index.row()]
        w = self._workers.get(worker_key, {})
        if role == QtCore.Qt.ItemDataRole.BackgroundRole:
            os_s = (w.get("os", "") or "").lower()
            # Windows — синий, Ubuntu/Kubuntu — зелёный, macOS — жёлтый, остальные — нейтральный серый.
            if "windows" in os_s:
                color = QtGui.QColor("#dbeafe")  # light blue
            elif "ubuntu" in os_s or "kubuntu" in os_s:
                color = QtGui.QColor("#dcfce7")  # light green
            elif "darwin" in os_s or "mac" in os_s or "macos" in os_s:
                color = QtGui.QColor("#fef9c3")  # light yellow
            elif "linux" in os_s:
                color = QtGui.QColor("#ecfccb")  # light lime
            else:
                color = QtGui.QColor("#f3f4f6")  # light gray
            return QtGui.QBrush(color)
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        col = index.column()
        if col == 0:
            return worker_key
        if col == 1:
            return w.get("status", "")
        if col == 2:
            ts = w.get("token_status", "") or ""
            return ts or "—"
        if col == 3:
            return w.get("os", "") or "—"
        if col == 4:
            return str(w.get("cpu_cores", ""))
        if col == 5:
            avail = w.get("ram_available_mb")
            total = w.get("ram_total_mb")
            if avail is not None and total is not None:
                return f"{avail} / {total}"
            return str(total or avail or "")
        if col == 6:
            gpus = w.get("gpus", [])
            return ", ".join(str(g.get("name", "")) for g in gpus) if gpus else "—"
        return ""

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return str(section)


def main() -> None:
    import argparse
    import atexit
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Chaboss Cluster GUI")
    parser.add_argument("--start-worker", action="store_true", help="Автоматически запустить воркера при старте GUI")
    args, _ = parser.parse_known_args()

    project_root = Path(__file__).resolve().parents[1]  # .../chaboss_cluster
    pid_file = project_root / ".chaboss_gui.pid"

    def write_pid() -> None:
        try:
            pid_file.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass

    def cleanup_pid() -> None:
        try:
            if pid_file.exists():
                # Удаляем только если PID наш (защита от гонок).
                cur = pid_file.read_text(encoding="utf-8").strip()
                if cur == str(os.getpid()):
                    pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    app = QtWidgets.QApplication(sys.argv)
    master_addr = (os.environ.get("CLUSTER_MASTER_ADDR") or "").strip() or None
    win = MainWindow(master_addr=master_addr)
    win.resize(900, 500)
    win.show()
    write_pid()
    app.aboutToQuit.connect(cleanup_pid)
    atexit.register(cleanup_pid)
    if args.start_worker:
        # Запуск после показа окна, чтобы UI успел инициализироваться.
        QtCore.QTimer.singleShot(0, win._on_start_worker)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
