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
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import re

from PyQt6 import QtCore, QtGui, QtWidgets

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
    worker_master_status = QtCore.pyqtSignal(str, str)  # (ключ воркера или "", статус в реестре мастера)

    def __init__(self, master_addr: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Chaboss Cluster")
        self.load_finished.connect(self._on_load_finished)
        self.load_progress_event.connect(self._on_load_progress_event)
        self.unload_finished.connect(self._on_unload_finished)
        self.apply_workers_finished.connect(self._on_apply_workers_finished)
        self.log_message.connect(self._append_log_line)
        self.worker_master_status.connect(self._on_worker_master_status)

        settings = load_settings()
        self._addr = (
            master_addr
            or os.environ.get("CLUSTER_MASTER_ADDR")
            or settings.get("master_addr", DEFAULT_MASTER_ADDR)
        ).strip()
        self._poller = MasterPoller(self._addr, self)
        self._last_workers_dict: Dict[str, dict] = {}
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._worker_visibility_timer = QtCore.QTimer(self)
        self._worker_visibility_timer.timeout.connect(self._poll_worker_visibility)
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_settings)

        self._table_model = WorkerTableModel()
        table = QtWidgets.QTableView()
        table.setModel(self._table_model)
        table.horizontalHeader().setStretchLastSection(True)

        self._addr_label = QtWidgets.QLabel(f"Мастер: {self._addr} (обновление каждые {POLL_INTERVAL_MS // 1000} с)")

        table_container = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self._addr_label)
        table_layout.addWidget(table)
        table_container.setLayout(table_layout)

        log_container = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QtWidgets.QLabel("Лог"))
        self._log_text = QtWidgets.QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setPlaceholderText(
            "Здесь отображаются процессы выгрузки/загрузки модели, статусы воркеров и прочие события."
        )
        log_layout.addWidget(self._log_text)
        clear_log_btn = QtWidgets.QPushButton("Очистить лог")
        clear_log_btn.clicked.connect(self._log_text.clear)
        log_layout.addWidget(clear_log_btn)
        log_container.setLayout(log_layout)

        # Панель «Режим воркера»: показывается при запущенном воркере вместо таблицы кластера
        self._worker_panel = QtWidgets.QWidget()
        worker_panel_layout = QtWidgets.QVBoxLayout()
        self._worker_status_label = QtWidgets.QLabel("Режим: Воркер\nПорт: —\nСтатус: не запущен")
        self._worker_status_label.setStyleSheet("font-size: 13px; padding: 12px;")
        worker_panel_layout.addWidget(self._worker_status_label)
        worker_panel_layout.addWidget(
            QtWidgets.QLabel("Мастер подключается к воркерам по своему конфигу. Этот узел ожидает запросы от мастера.")
        )
        worker_panel_layout.addStretch()
        self._worker_panel.setLayout(worker_panel_layout)

        self._worker_master_status_text = ""  # строка для подстановки в панель воркера
        self._cluster_stack = QtWidgets.QStackedWidget()
        self._cluster_stack.addWidget(table_container)   # индекс 0: вид «мастер» (таблица воркеров)
        self._cluster_stack.addWidget(self._worker_panel)  # индекс 1: вид «воркер»

        splitter_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter_main.addWidget(self._cluster_stack)
        splitter_main.addWidget(log_container)
        splitter_main.setStretchFactor(0, 7)
        splitter_main.setStretchFactor(1, 3)

        workers_layout = QtWidgets.QVBoxLayout()
        workers_layout.addWidget(splitter_main)
        workers_page = QtWidgets.QWidget()
        workers_page.setLayout(workers_layout)

        # ——— Вкладка «Настройки» ———
        settings_layout = QtWidgets.QVBoxLayout()

        # Адрес мастера и запуск мастера
        master_grp = QtWidgets.QGroupBox("Подключение к мастеру")
        master_layout = QtWidgets.QVBoxLayout()
        self._master_edit = QtWidgets.QLineEdit()
        self._master_edit.setPlaceholderText("host:port, например 127.0.0.1:60051")
        self._master_edit.setText(self._addr)
        self._master_edit.textChanged.connect(self._schedule_save)
        master_layout.addWidget(self._master_edit)
        master_btn_layout = QtWidgets.QHBoxLayout()
        start_master_btn = QtWidgets.QPushButton("Старт мастера")
        start_master_btn.setToolTip("Запуск мастера в отдельном процессе (python -m scripts.run_master)")
        start_master_btn.clicked.connect(self._on_start_master)
        stop_master_btn = QtWidgets.QPushButton("Остановить мастера")
        stop_master_btn.clicked.connect(self._on_stop_master)
        restart_master_btn = QtWidgets.QPushButton("Перезапустить мастера")
        restart_master_btn.clicked.connect(self._on_restart_master)
        master_btn_layout.addWidget(start_master_btn)
        master_btn_layout.addWidget(stop_master_btn)
        master_btn_layout.addWidget(restart_master_btn)
        master_layout.addLayout(master_btn_layout)
        self._master_process: subprocess.Popen | None = None
        self._start_master_btn = start_master_btn
        self._stop_master_btn = stop_master_btn
        self._restart_master_btn = restart_master_btn
        self._master_grp = master_grp
        master_grp.setLayout(master_layout)
        settings_layout.addWidget(master_grp)

        # Режим загрузки модели и использование ресурсов
        mode_grp = QtWidgets.QGroupBox("Режим загрузки модели")
        mode_layout = QtWidgets.QVBoxLayout()
        self._mode_combo = QtWidgets.QComboBox()
        for value, label in MODEL_LOAD_MODES:
            self._mode_combo.addItem(label, value)
        self._mode_combo.currentIndexChanged.connect(self._schedule_save)
        mode_layout.addWidget(QtWidgets.QLabel("Режим:"))
        mode_layout.addWidget(self._mode_combo)
        resource_layout = QtWidgets.QHBoxLayout()
        resource_layout.addWidget(QtWidgets.QLabel("Использование свободных ресурсов (%):"))
        self._resource_percent_spin = QtWidgets.QSpinBox()
        self._resource_percent_spin.setRange(1, 100)
        self._resource_percent_spin.setValue(75)
        self._resource_percent_spin.setSuffix(" %")
        self._resource_percent_spin.valueChanged.connect(self._schedule_save)
        self._resource_percent_spin.valueChanged.connect(
            lambda: self._update_resources_label(getattr(self, "_last_workers_dict", {}))
        )
        resource_layout.addWidget(self._resource_percent_spin)
        resource_layout.addStretch()
        mode_layout.addLayout(resource_layout)
        self._resources_label = QtWidgets.QLabel("Свободные ресурсы: — (выполните Скан на вкладке Воркеры)")
        self._resources_label.setWordWrap(True)
        mode_layout.addWidget(self._resources_label)
        self._mode_grp = mode_grp
        mode_grp.setLayout(mode_layout)
        settings_layout.addWidget(mode_grp)

        # Конфиг воркеров (IP, port, key)
        workers_grp = QtWidgets.QGroupBox("Конфиг воркеров")
        w_layout = QtWidgets.QVBoxLayout()
        self._workers_table = QtWidgets.QTableWidget(0, 3)
        self._workers_table.setHorizontalHeaderLabels(["IP (host)", "Port", "Key (auth_token)"])
        self._workers_table.horizontalHeader().setStretchLastSection(True)
        self._workers_table.cellChanged.connect(self._schedule_save)
        w_btn_layout = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Добавить")
        remove_btn = QtWidgets.QPushButton("Удалить")
        apply_btn = QtWidgets.QPushButton("Применить конфиг на мастере")
        add_btn.clicked.connect(self._add_worker_row)
        remove_btn.clicked.connect(self._remove_worker_row)
        apply_btn.clicked.connect(self._on_apply_workers_config)
        w_btn_layout.addWidget(add_btn)
        w_btn_layout.addWidget(remove_btn)
        w_btn_layout.addWidget(apply_btn)
        w_btn_layout.addStretch()
        w_layout.addWidget(self._workers_table)
        w_layout.addLayout(w_btn_layout)
        self._workers_grp = workers_grp
        workers_grp.setLayout(w_layout)
        settings_layout.addWidget(workers_grp)

        # Модель HF и кнопки
        model_grp = QtWidgets.QGroupBox("Модель и запуск")
        model_layout = QtWidgets.QVBoxLayout()
        self._model_edit = QtWidgets.QLineEdit()
        self._model_edit.setPlaceholderText("например: bert-base-uncased или org/repo")
        self._model_edit.textChanged.connect(self._schedule_save)
        scan_btn = QtWidgets.QPushButton("Скан")
        start_btn = QtWidgets.QPushButton("Старт")
        unload_btn = QtWidgets.QPushButton("Выгрузить модель")
        scan_btn.clicked.connect(self._on_scan)
        start_btn.clicked.connect(self._on_start)
        unload_btn.clicked.connect(self._on_unload_model)
        model_layout.addWidget(QtWidgets.QLabel("Модель HuggingFace:"))
        model_layout.addWidget(self._model_edit)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(scan_btn)
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(unload_btn)
        btn_layout.addStretch()
        model_layout.addLayout(btn_layout)
        self._model_grp = model_grp
        model_grp.setLayout(model_layout)
        settings_layout.addWidget(model_grp)

        settings_layout.addStretch()
        settings_page = QtWidgets.QWidget()
        settings_page.setLayout(settings_layout)

        # ——— Вкладка «Воркер» ———
        worker_tab = QtWidgets.QWidget()
        worker_tab_layout = QtWidgets.QVBoxLayout()
        worker_cfg_grp = QtWidgets.QGroupBox("Запуск воркера")
        worker_cfg_layout = QtWidgets.QVBoxLayout()
        worker_cfg_layout.addWidget(QtWidgets.QLabel("Конфиг воркера (путь относительно корня проекта):"))
        self._worker_config_edit = QtWidgets.QLineEdit()
        self._worker_config_edit.setPlaceholderText("config/worker.yaml")
        self._worker_config_edit.textChanged.connect(self._schedule_save)
        worker_cfg_layout.addWidget(self._worker_config_edit)
        worker_btn_layout = QtWidgets.QHBoxLayout()
        self._start_worker_btn = QtWidgets.QPushButton("Старт воркера")
        self._start_worker_btn.clicked.connect(self._on_start_worker)
        self._stop_worker_btn = QtWidgets.QPushButton("Остановить воркера")
        self._stop_worker_btn.clicked.connect(self._on_stop_worker)
        self._restart_worker_btn = QtWidgets.QPushButton("Перезапустить воркера")
        self._restart_worker_btn.clicked.connect(self._on_restart_worker)
        worker_btn_layout.addWidget(self._start_worker_btn)
        worker_btn_layout.addWidget(self._stop_worker_btn)
        worker_btn_layout.addWidget(self._restart_worker_btn)
        worker_btn_layout.addStretch()
        worker_cfg_layout.addLayout(worker_btn_layout)
        worker_cfg_grp.setLayout(worker_cfg_layout)
        worker_tab_layout.addWidget(worker_cfg_grp)
        worker_tab_layout.addStretch()
        worker_tab.setLayout(worker_tab_layout)
        self._worker_process: subprocess.Popen | None = None
        self._worker_display_port: int | None = None

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(workers_page, "Кластер")
        tabs.addTab(settings_page, "Настройки")
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
                suffix = ":%s" % port
                for key, w in workers.items():
                    if key.endswith(suffix):
                        status = w.get("status", "UNKNOWN")
                        self.worker_master_status.emit(key, status)
                        return
                self.worker_master_status.emit("", "не найден")
            except Exception as e:
                self.worker_master_status.emit("", "ошибка: %s" % (e if len(str(e)) < 60 else str(e)[:57] + "..."))

        threading.Thread(target=do_poll, daemon=True).start()

    def _on_worker_master_status(self, worker_key: str, status: str) -> None:
        """Обновить строку «В реестре мастера» и перерисовать панель воркера."""
        if worker_key:
            self._worker_master_status_text = "В реестре мастера: %s (%s)" % (status, worker_key)
        else:
            self._worker_master_status_text = "В реестре мастера: %s" % status
        if getattr(self, "_worker_process", None) is not None and self._worker_process.poll() is None:
            self._refresh_worker_status_label()

    def _update_process_state(self) -> None:
        """Обновить вид кластера и блокировки: мастер и воркер взаимоисключающие."""
        master_running = self._master_process is not None and self._master_process.poll() is None
        worker_running = self._worker_process is not None and self._worker_process.poll() is None

        if worker_running:
            self._cluster_stack.setCurrentIndex(1)
            self._worker_master_status_text = "В реестре мастера: проверка..."
            self._refresh_worker_status_label()
            self._timer.stop()
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
        try:
            if path_resolved.exists():
                with open(path_resolved, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
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
            try:
                channel = grpc.insecure_channel(self._addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                try:
                    stream = stub.LoadModelStream(
                        cluster_pb2.LoadModelRequest(hf_model_id=model_id),
                    )
                    for ev in stream:
                        self.load_progress_event.emit(_ev_to_dict(ev))
                        if ev.done:
                            self.load_finished.emit(ev.ok, ev.error or "")
                            break
                except grpc.RpcError as rpc_err:
                    if rpc_err.code() == grpc.StatusCode.UNIMPLEMENTED or "Method not found" in (rpc_err.details() or ""):
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
                self.load_finished.emit(False, str(e))

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
        msg = (
            "Ошибка загрузки модели:\n\n%s\n\n"
            "Подробные причины смотрите в логах мастера и воркеров "
            "(консоль, где запущены master/worker, или журнал приложения).\n\n"
            "%s"
            "Рекомендации:\n"
            "• Если модель не влезает в выбранный %% свободных ресурсов — увеличьте "
            "«Использование свободных ресурсов (%%)» в настройках (например, до 80–90%%).\n"
            "• Если модель не влезает даже при 100%%:\n"
            "  (a) Переключитесь в режим «Модель по частям (стриминг)»;\n"
            "  (b) Добавьте воркеров в кластер для увеличения суммарных ресурсов."
        ) % (error, deadline_hint)
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
            self._load_worker_bars[key].setValue(min(100, prog.get("percent", 0)))
        if ev.get("done"):
            if ev.get("ok"):
                self._load_stage_label.setText("Модель успешно загружена: мастер и воркеры готовы")
            else:
                self._load_stage_label.setText("Ошибка при загрузке модели — см. сообщение ниже")

    def _on_load_finished(self, ok: bool, error: str) -> None:
        self._model_edit.setEnabled(True)
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
        self.statusBar().showMessage(f"Ошибка: {msg}")
        self._table_model.update_workers({})
        self._last_workers_dict = {}


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
    app = QtWidgets.QApplication(sys.argv)
    master_addr = (os.environ.get("CLUSTER_MASTER_ADDR") or "").strip() or None
    win = MainWindow(master_addr=master_addr)
    win.resize(900, 500)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
