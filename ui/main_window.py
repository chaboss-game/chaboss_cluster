from __future__ import annotations

import sys
from typing import Dict

from PyQt6 import QtCore, QtWidgets


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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cluster Master UI")

        self._table_model = WorkerTableModel()
        table = QtWidgets.QTableView()
        table.setModel(self._table_model)
        table.horizontalHeader().setStretchLastSection(True)

        self.setCentralWidget(table)

        # TODO: periodically poll master node for worker status via admin API.


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 400)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

