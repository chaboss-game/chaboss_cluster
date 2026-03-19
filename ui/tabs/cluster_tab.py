from __future__ import annotations

from typing import Any

from PyQt6 import QtCore, QtWidgets


class ClusterTabWidget:
    """Builds the 'Cluster' tab UI and binds required owner attributes."""

    def __init__(self, owner: Any, table_model: QtCore.QAbstractTableModel, poll_interval_ms: int) -> None:
        self._owner = owner
        self._table_model = table_model
        self._poll_interval_ms = poll_interval_ms

    def build(self) -> QtWidgets.QWidget:
        owner = self._owner

        table = QtWidgets.QTableView()
        table.setModel(self._table_model)
        table.horizontalHeader().setStretchLastSection(True)

        owner._addr_label = QtWidgets.QLabel(
            f"Мастер: {owner._addr} (обновление каждые {self._poll_interval_ms // 1000} с)"
        )

        table_container = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(owner._addr_label)
        table_layout.addWidget(table)
        table_container.setLayout(table_layout)

        log_container = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QtWidgets.QLabel("Лог"))
        owner._log_text = QtWidgets.QPlainTextEdit()
        owner._log_text.setReadOnly(True)
        owner._log_text.setPlaceholderText(
            "События GUI, логи мастера и воркеров (то же, что пишется в файлы на соответствующих узлах)."
        )
        log_layout.addWidget(owner._log_text)
        clear_log_btn = QtWidgets.QPushButton("Очистить лог")
        clear_log_btn.clicked.connect(owner._log_text.clear)
        log_layout.addWidget(clear_log_btn)
        log_container.setLayout(log_layout)

        owner._worker_panel = QtWidgets.QWidget()
        worker_panel_layout = QtWidgets.QVBoxLayout()
        owner._worker_status_label = QtWidgets.QLabel("Режим: Воркер\nПорт: —\nСтатус: не запущен")
        owner._worker_status_label.setStyleSheet("font-size: 13px; padding: 12px;")
        worker_panel_layout.addWidget(owner._worker_status_label)
        worker_panel_layout.addWidget(
            QtWidgets.QLabel(
                "Мастер подключается к воркерам по своему конфигу. Этот узел ожидает запросы от мастера."
            )
        )
        worker_panel_layout.addStretch()
        owner._worker_panel.setLayout(worker_panel_layout)

        owner._worker_master_status_text = ""
        owner._cluster_stack = QtWidgets.QStackedWidget()
        owner._cluster_stack.addWidget(table_container)
        owner._cluster_stack.addWidget(owner._worker_panel)

        splitter_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter_main.addWidget(owner._cluster_stack)
        splitter_main.addWidget(log_container)
        splitter_main.setStretchFactor(0, 7)
        splitter_main.setStretchFactor(1, 3)

        workers_layout = QtWidgets.QVBoxLayout()
        workers_layout.addWidget(splitter_main)
        workers_page = QtWidgets.QWidget()
        workers_page.setLayout(workers_layout)
        return workers_page
