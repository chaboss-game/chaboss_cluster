"""
PyQt6 UI для мониторинга кластера.
Опрашивает мастер по gRPC (ListWorkers) и отображает таблицу воркеров.
Адрес мастера: переменная окружения CLUSTER_MASTER_ADDR или 127.0.0.1:60051.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Dict

from PyQt6 import QtCore, QtWidgets

# Импорт gRPC после установки зависимостей и запуска scripts/gen_grpc.py
try:
    import grpc
    from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc
except ImportError as e:
    print(
        "Ошибка: сгенерируйте gRPC-модули: pip install -r requirements.txt && python scripts/gen_grpc.py",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

# Адрес мастера: host:port
DEFAULT_MASTER_ADDR = "127.0.0.1:60051"
POLL_INTERVAL_MS = 3000

_STATUS_NAMES = {
    0: "ONLINE",
    1: "OFFLINE",
    2: "UNSTABLE",
    3: "RECONNECTING",
}


def _worker_list_to_dict(worker_list) -> Dict[str, dict]:
    """Преобразует proto WorkerList в словарь для таблицы."""
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
    """Фоновый опрос мастера; результат передаётся в UI по сигналу."""

    workers_updated = QtCore.pyqtSignal(dict)  # worker_key -> worker dict
    error = QtCore.pyqtSignal(str)

    def __init__(self, master_addr: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._master_addr = master_addr

    def poll(self) -> None:
        try:
            channel = grpc.insecure_channel(self._master_addr)
            stub = cluster_pb2_grpc.MasterAdminServiceStub(channel)
            response = stub.ListWorkers(cluster_pb2.Empty(), timeout=5.0)
            channel.close()
            self.workers_updated.emit(_worker_list_to_dict(response))
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, master_addr: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Cluster Master UI")
        addr = master_addr or os.environ.get("CLUSTER_MASTER_ADDR", DEFAULT_MASTER_ADDR)
        self._poller = MasterPoller(addr, self)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._table_model = WorkerTableModel()
        table = QtWidgets.QTableView()
        table.setModel(self._table_model)
        table.horizontalHeader().setStretchLastSection(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(f"Мастер: {addr} (обновление каждые {POLL_INTERVAL_MS // 1000} с)"))
        layout.addWidget(table)
        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Загрузка...")

        self._poller.workers_updated.connect(self._on_workers_updated)
        self._poller.error.connect(self._on_poller_error)
        self._timer.start(POLL_INTERVAL_MS)
        self._on_timer()

    def _on_timer(self) -> None:
        threading.Thread(target=self._poller.poll, daemon=True).start()

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

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self._keys)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self.HEADERS)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
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

    def headerData(  # type: ignore[override]
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return str(section)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    master_addr = os.environ.get("CLUSTER_MASTER_ADDR", DEFAULT_MASTER_ADDR).strip() or None
    win = MainWindow(master_addr=master_addr)
    win.resize(900, 400)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
