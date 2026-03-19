from __future__ import annotations

import subprocess
from typing import Any

from PyQt6 import QtWidgets


class SettingsTabWidget:
    """Builds the 'Settings' tab UI and binds required owner attributes."""

    def __init__(self, owner: Any, model_load_modes: list[tuple[str, str]]) -> None:
        self._owner = owner
        self._model_load_modes = model_load_modes

    def build(self) -> QtWidgets.QWidget:
        owner = self._owner
        settings_layout = QtWidgets.QVBoxLayout()

        master_grp = QtWidgets.QGroupBox("Подключение к мастеру")
        master_layout = QtWidgets.QVBoxLayout()
        owner._master_edit = QtWidgets.QLineEdit()
        owner._master_edit.setPlaceholderText("host:port, например 127.0.0.1:60051")
        owner._master_edit.setText(owner._addr)
        owner._master_edit.textChanged.connect(owner._schedule_save)
        master_layout.addWidget(owner._master_edit)
        master_btn_layout = QtWidgets.QHBoxLayout()
        start_master_btn = QtWidgets.QPushButton("Старт мастера")
        start_master_btn.setToolTip("Запуск мастера в отдельном процессе (python -m scripts.run_master)")
        start_master_btn.clicked.connect(owner._on_start_master)
        stop_master_btn = QtWidgets.QPushButton("Остановить мастера")
        stop_master_btn.clicked.connect(owner._on_stop_master)
        restart_master_btn = QtWidgets.QPushButton("Перезапустить мастера")
        restart_master_btn.clicked.connect(owner._on_restart_master)
        master_btn_layout.addWidget(start_master_btn)
        master_btn_layout.addWidget(stop_master_btn)
        master_btn_layout.addWidget(restart_master_btn)
        master_layout.addLayout(master_btn_layout)
        owner._master_process: subprocess.Popen | None = None
        owner._start_master_btn = start_master_btn
        owner._stop_master_btn = stop_master_btn
        owner._restart_master_btn = restart_master_btn
        owner._master_grp = master_grp
        master_grp.setLayout(master_layout)
        settings_layout.addWidget(master_grp)

        mode_grp = QtWidgets.QGroupBox("Режим загрузки модели")
        mode_layout = QtWidgets.QVBoxLayout()
        owner._mode_combo = QtWidgets.QComboBox()
        for value, label in self._model_load_modes:
            owner._mode_combo.addItem(label, value)
        owner._mode_combo.currentIndexChanged.connect(owner._schedule_save)
        mode_layout.addWidget(QtWidgets.QLabel("Режим:"))
        mode_layout.addWidget(owner._mode_combo)
        resource_layout = QtWidgets.QHBoxLayout()
        resource_layout.addWidget(QtWidgets.QLabel("Использование свободных ресурсов (%):"))
        owner._resource_percent_spin = QtWidgets.QSpinBox()
        owner._resource_percent_spin.setRange(1, 100)
        owner._resource_percent_spin.setValue(75)
        owner._resource_percent_spin.setSuffix(" %")
        owner._resource_percent_spin.valueChanged.connect(owner._schedule_save)
        owner._resource_percent_spin.valueChanged.connect(
            lambda: owner._update_resources_label(getattr(owner, "_last_workers_dict", {}))
        )
        resource_layout.addWidget(owner._resource_percent_spin)
        resource_layout.addStretch()
        mode_layout.addLayout(resource_layout)
        owner._resources_label = QtWidgets.QLabel("Свободные ресурсы: — (выполните Скан на вкладке Воркеры)")
        owner._resources_label.setWordWrap(True)
        mode_layout.addWidget(owner._resources_label)
        owner._mode_grp = mode_grp
        mode_grp.setLayout(mode_layout)
        settings_layout.addWidget(mode_grp)

        workers_grp = QtWidgets.QGroupBox("Конфиг воркеров")
        w_layout = QtWidgets.QVBoxLayout()
        owner._workers_table = QtWidgets.QTableWidget(0, 3)
        owner._workers_table.setHorizontalHeaderLabels(["IP (host)", "Port", "Key (auth_token)"])
        owner._workers_table.horizontalHeader().setStretchLastSection(True)
        owner._workers_table.cellChanged.connect(owner._schedule_save)
        w_btn_layout = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Добавить")
        remove_btn = QtWidgets.QPushButton("Удалить")
        apply_btn = QtWidgets.QPushButton("Применить конфиг на мастере")
        update_btn = QtWidgets.QPushButton("Обновить воркеры (git pull)")
        add_btn.clicked.connect(owner._add_worker_row)
        remove_btn.clicked.connect(owner._remove_worker_row)
        apply_btn.clicked.connect(owner._on_apply_workers_config)
        update_btn.clicked.connect(owner._on_remote_update_workers)
        w_btn_layout.addWidget(add_btn)
        w_btn_layout.addWidget(remove_btn)
        w_btn_layout.addWidget(apply_btn)
        w_btn_layout.addWidget(update_btn)
        w_btn_layout.addStretch()
        w_layout.addWidget(owner._workers_table)
        w_layout.addLayout(w_btn_layout)
        owner._workers_grp = workers_grp
        workers_grp.setLayout(w_layout)
        settings_layout.addWidget(workers_grp)

        model_grp = QtWidgets.QGroupBox("Модель и запуск")
        model_layout = QtWidgets.QVBoxLayout()
        owner._model_edit = QtWidgets.QLineEdit()
        owner._model_edit.setPlaceholderText("например: bert-base-uncased или org/repo")
        owner._model_edit.textChanged.connect(owner._schedule_save)
        scan_btn = QtWidgets.QPushButton("Скан")
        start_btn = QtWidgets.QPushButton("Старт")
        unload_btn = QtWidgets.QPushButton("Выгрузить модель")
        scan_btn.clicked.connect(owner._on_scan)
        start_btn.clicked.connect(owner._on_start)
        unload_btn.clicked.connect(owner._on_unload_model)
        model_layout.addWidget(QtWidgets.QLabel("Модель HuggingFace:"))
        model_layout.addWidget(owner._model_edit)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(scan_btn)
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(unload_btn)
        btn_layout.addStretch()
        model_layout.addLayout(btn_layout)
        owner._model_grp = model_grp
        model_grp.setLayout(model_layout)
        settings_layout.addWidget(model_grp)

        settings_layout.addStretch()
        settings_page = QtWidgets.QWidget()
        settings_page.setLayout(settings_layout)
        return settings_page
