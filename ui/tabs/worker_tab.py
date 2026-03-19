from __future__ import annotations

import subprocess
from typing import Any

from PyQt6 import QtWidgets


class WorkerTabWidget:
    """Builds the 'Worker' tab UI and binds required owner attributes."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def build(self) -> QtWidgets.QWidget:
        owner = self._owner
        worker_tab = QtWidgets.QWidget()
        worker_tab_layout = QtWidgets.QVBoxLayout()
        worker_cfg_grp = QtWidgets.QGroupBox("Запуск воркера")
        worker_cfg_layout = QtWidgets.QVBoxLayout()
        worker_cfg_layout.addWidget(
            QtWidgets.QLabel("Конфиг воркера (путь относительно корня проекта):")
        )
        owner._worker_config_edit = QtWidgets.QLineEdit()
        owner._worker_config_edit.setPlaceholderText("config/worker.yaml")
        owner._worker_config_edit.textChanged.connect(owner._schedule_save)
        worker_cfg_layout.addWidget(owner._worker_config_edit)
        worker_btn_layout = QtWidgets.QHBoxLayout()
        owner._start_worker_btn = QtWidgets.QPushButton("Старт воркера")
        owner._start_worker_btn.clicked.connect(owner._on_start_worker)
        owner._stop_worker_btn = QtWidgets.QPushButton("Остановить воркера")
        owner._stop_worker_btn.clicked.connect(owner._on_stop_worker)
        owner._restart_worker_btn = QtWidgets.QPushButton("Перезапустить воркера")
        owner._restart_worker_btn.clicked.connect(owner._on_restart_worker)
        worker_btn_layout.addWidget(owner._start_worker_btn)
        worker_btn_layout.addWidget(owner._stop_worker_btn)
        worker_btn_layout.addWidget(owner._restart_worker_btn)
        worker_btn_layout.addStretch()
        worker_cfg_layout.addLayout(worker_btn_layout)
        worker_cfg_grp.setLayout(worker_cfg_layout)
        worker_tab_layout.addWidget(worker_cfg_grp)
        worker_tab_layout.addStretch()
        worker_tab.setLayout(worker_tab_layout)
        owner._worker_process: subprocess.Popen | None = None
        owner._worker_display_port: int | None = None
        return worker_tab
