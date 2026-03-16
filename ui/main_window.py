"""
PyQt6 UI для мониторинга кластера.
Вкладки: Воркеры, Настройки (адрес мастера, конфиг воркеров IP/port/key, модель HF, Скан, Старт).
Все настройки автосохраняются.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Dict, List

from PyQt6 import QtCore, QtWidgets

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

DEFAULT_MASTER_ADDR = "127.0.0.1:60051"
POLL_INTERVAL_MS = 3000
AUTOSAVE_DELAY_MS = 500

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
            "cpu_cores": w.resources.cpu_cores,
            "ram_total_mb": w.resources.ram_total_mb,
            "gpus": [{"name": g.name} for g in w.resources.gpus],
        }
    return out


class MasterPoller(QtCore.QObject):
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


class MainWindow(QtWidgets.QMainWindow):
    load_finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, master_addr: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Cluster Master UI")
        self.load_finished.connect(self._on_load_finished)

        settings = load_settings()
        self._addr = (
            master_addr
            or os.environ.get("CLUSTER_MASTER_ADDR")
            or settings.get("master_addr", DEFAULT_MASTER_ADDR)
        ).strip()
        self._poller = MasterPoller(self._addr, self)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_settings)

        self._table_model = WorkerTableModel()
        table = QtWidgets.QTableView()
        table.setModel(self._table_model)
        table.horizontalHeader().setStretchLastSection(True)

        workers_layout = QtWidgets.QVBoxLayout()
        self._addr_label = QtWidgets.QLabel(f"Мастер: {self._addr} (обновление каждые {POLL_INTERVAL_MS // 1000} с)")
        workers_layout.addWidget(self._addr_label)
        workers_layout.addWidget(table)
        workers_page = QtWidgets.QWidget()
        workers_page.setLayout(workers_layout)

        # ——— Вкладка «Настройки» ———
        settings_layout = QtWidgets.QVBoxLayout()

        # Адрес мастера
        master_grp = QtWidgets.QGroupBox("Подключение к мастеру")
        master_layout = QtWidgets.QVBoxLayout()
        self._master_edit = QtWidgets.QLineEdit()
        self._master_edit.setPlaceholderText("host:port, например 127.0.0.1:60051")
        self._master_edit.setText(self._addr)
        self._master_edit.textChanged.connect(self._schedule_save)
        master_layout.addWidget(self._master_edit)
        master_grp.setLayout(master_layout)
        settings_layout.addWidget(master_grp)

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
        add_btn.clicked.connect(self._add_worker_row)
        remove_btn.clicked.connect(self._remove_worker_row)
        w_btn_layout.addWidget(add_btn)
        w_btn_layout.addWidget(remove_btn)
        w_btn_layout.addStretch()
        w_layout.addWidget(self._workers_table)
        w_layout.addLayout(w_btn_layout)
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
        scan_btn.clicked.connect(self._on_scan)
        start_btn.clicked.connect(self._on_start)
        model_layout.addWidget(QtWidgets.QLabel("Модель HuggingFace:"))
        model_layout.addWidget(self._model_edit)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(scan_btn)
        btn_layout.addWidget(start_btn)
        btn_layout.addStretch()
        model_layout.addLayout(btn_layout)
        model_grp.setLayout(model_layout)
        settings_layout.addWidget(model_grp)

        settings_layout.addStretch()
        settings_page = QtWidgets.QWidget()
        settings_page.setLayout(settings_layout)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(workers_page, "Воркеры")
        tabs.addTab(settings_page, "Настройки")

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Загрузка...")

        self._poller.workers_updated.connect(self._on_workers_updated)
        self._poller.error.connect(self._on_poller_error)
        self._timer.start(POLL_INTERVAL_MS)
        self._on_timer()

        # Применить загруженные настройки (модель, таблица воркеров)
        self._model_edit.setText(settings.get("hf_model_id", "") or "")
        self._load_workers_into_table(settings.get("workers", []))

        # Обновлять адрес поллера при переключении на вкладку Воркеры или по таймеру
        self._apply_master_addr_from_edit()

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
            "workers": self._get_workers_from_table(),
        }
        save_settings(data)

    def _apply_master_addr_from_edit(self) -> None:
        addr = self._master_edit.text().strip() or DEFAULT_MASTER_ADDR
        if addr != self._addr:
            self._addr = addr
            self._poller.set_master_addr(addr)
            self._addr_label.setText(f"Мастер: {addr} (обновление каждые {POLL_INTERVAL_MS // 1000} с)")

    def _on_timer(self) -> None:
        threading.Thread(target=self._poller.poll, daemon=True).start()

    def _on_scan(self) -> None:
        self._apply_master_addr_from_edit()
        self._save_settings()
        threading.Thread(target=self._poller.poll, daemon=True).start()
        self.statusBar().showMessage("Скан...")

    def _on_start(self) -> None:
        self._apply_master_addr_from_edit()
        self._save_settings()
        model_id = self._model_edit.text().strip()
        if not model_id:
            self.statusBar().showMessage("Введите название модели HuggingFace")
            return
        self.statusBar().showMessage(f"Загрузка модели {model_id}...")
        self._model_edit.setEnabled(False)

        def do_load():
            try:
                channel = grpc.insecure_channel(self._addr)
                stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
                resp = stub.LoadModel(
                    cluster_pb2.LoadModelRequest(hf_model_id=model_id),
                    timeout=300.0,
                )
                channel.close()
                self.load_finished.emit(resp.ok, resp.error or "")
            except Exception as e:
                self.load_finished.emit(False, str(e))

        threading.Thread(target=do_load, daemon=True).start()

    def _on_load_finished(self, ok: bool, error: str) -> None:
        self._model_edit.setEnabled(True)
        if ok:
            self.statusBar().showMessage("Модель загружена и разослана воркерам")
        else:
            self.statusBar().showMessage(f"Ошибка: {error}")

    def _on_workers_updated(self, workers: Dict[str, dict]) -> None:
        self._table_model.update_workers(workers)
        self.statusBar().showMessage(f"Воркеров: {len(workers)}")

    def _on_poller_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Ошибка: {msg}")
        self._table_model.update_workers({})


class WorkerTableModel(QtCore.QAbstractTableModel):
    HEADERS = ["ID", "Status", "CPU", "RAM (MB)", "GPUs"]

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
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        worker_key = self._keys[index.row()]
        w = self._workers.get(worker_key, {})
        col = index.column()
        if col == 0:
            return worker_key
        if col == 1:
            return w.get("status", "")
        if col == 2:
            return str(w.get("cpu_cores", ""))
        if col == 3:
            return str(w.get("ram_total_mb", ""))
        if col == 4:
            return ", ".join(str(g.get("name")) for g in w.get("gpus", []))
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
