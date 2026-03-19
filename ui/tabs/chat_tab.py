from __future__ import annotations

from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets


class ChatTabWidget:
    """Builds the 'Chat' tab UI and binds required owner attributes."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def build(self) -> QtWidgets.QWidget:
        owner = self._owner
        chat_tab = QtWidgets.QWidget()
        chat_tab_layout = QtWidgets.QVBoxLayout()
        chat_tab_layout.setContentsMargins(8, 8, 8, 8)
        chat_tab_layout.setSpacing(8)

        chat_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        chat_channels_panel = QtWidgets.QWidget()
        chat_channels_layout = QtWidgets.QVBoxLayout()
        chat_channels_layout.addWidget(QtWidgets.QLabel("Канал:"))
        owner._chat_channels_combo = QtWidgets.QComboBox()
        owner._chat_channels_combo.currentIndexChanged.connect(owner._chat_on_channel_changed)
        chat_channels_layout.addWidget(owner._chat_channels_combo)

        chat_channels_btns = QtWidgets.QHBoxLayout()
        add_channel_btn = QtWidgets.QPushButton("Создать")
        rename_channel_btn = QtWidgets.QPushButton("Переименовать")
        del_channel_btn = QtWidgets.QPushButton("Удалить")
        add_channel_btn.clicked.connect(owner._chat_create_channel)
        rename_channel_btn.clicked.connect(owner._chat_rename_channel)
        del_channel_btn.clicked.connect(owner._chat_delete_channel)
        chat_channels_btns.addWidget(add_channel_btn)
        chat_channels_btns.addWidget(rename_channel_btn)
        chat_channels_btns.addWidget(del_channel_btn)
        chat_channels_layout.addLayout(chat_channels_btns)
        recv_row = QtWidgets.QHBoxLayout()
        recv_row.setContentsMargins(0, 0, 0, 0)
        recv_row.setSpacing(6)
        recv_row.addWidget(QtWidgets.QLabel("Получатели (онлайн):"))
        owner._chat_refresh_receivers_btn = QtWidgets.QToolButton()
        owner._chat_refresh_receivers_btn.setToolTip("Обновить получателей")
        owner._chat_refresh_receivers_btn.setEnabled(True)
        owner._chat_refresh_receivers_btn.setMinimumWidth(30)
        owner._chat_refresh_receivers_btn.setMaximumWidth(30)
        owner._chat_refresh_receivers_btn.setIcon(
            owner.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
        )
        owner._chat_refresh_receivers_btn.clicked.connect(owner._chat_refresh_channels_and_receivers)
        recv_row.addWidget(owner._chat_refresh_receivers_btn, 0)
        chat_channels_layout.addLayout(recv_row)
        owner._chat_workers_checklist = QtWidgets.QListWidget()
        owner._chat_workers_checklist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        owner._chat_workers_checklist.setMinimumHeight(80)
        owner._chat_workers_checklist.setMaximumHeight(180)
        chat_channels_layout.addWidget(owner._chat_workers_checklist)
        chat_channels_layout.addStretch()
        chat_channels_panel.setLayout(chat_channels_layout)

        chat_messages_panel = QtWidgets.QWidget()
        chat_messages_layout = QtWidgets.QVBoxLayout()
        chat_messages_layout.addWidget(QtWidgets.QLabel("Сообщения:"))
        owner._chat_messages_list = QtWidgets.QListWidget()
        owner._chat_messages_list.currentRowChanged.connect(owner._chat_on_message_selected)
        chat_messages_layout.addWidget(owner._chat_messages_list, 1)
        chat_messages_panel.setLayout(chat_messages_layout)

        chat_attach_panel = QtWidgets.QWidget()
        chat_attach_layout = QtWidgets.QVBoxLayout()
        chat_attach_layout.addWidget(QtWidgets.QLabel("Вложения:"))
        owner._chat_attachments_scroll = QtWidgets.QScrollArea()
        owner._chat_attachments_scroll.setWidgetResizable(True)
        owner._chat_attachments_container = QtWidgets.QWidget()
        owner._chat_attachments_layout = QtWidgets.QVBoxLayout()
        owner._chat_attachments_layout.setContentsMargins(6, 6, 6, 6)
        owner._chat_attachments_container.setLayout(owner._chat_attachments_layout)
        owner._chat_attachments_scroll.setWidget(owner._chat_attachments_container)
        chat_attach_layout.addWidget(owner._chat_attachments_scroll, 1)
        chat_attach_panel.setLayout(chat_attach_layout)

        chat_split.addWidget(chat_channels_panel)
        chat_split.addWidget(chat_messages_panel)
        chat_split.addWidget(chat_attach_panel)
        chat_split.setStretchFactor(0, 1)
        chat_split.setStretchFactor(1, 3)
        chat_split.setStretchFactor(2, 2)
        chat_split.setChildrenCollapsible(False)
        chat_tab_layout.addWidget(chat_split, 2)

        send_grp = QtWidgets.QGroupBox("Отправка")
        send_box_layout = QtWidgets.QVBoxLayout()
        send_box_layout.setContentsMargins(10, 10, 10, 10)
        send_box_layout.setSpacing(8)

        send_layout = QtWidgets.QHBoxLayout()
        send_layout.setContentsMargins(0, 0, 0, 0)
        send_layout.setSpacing(12)

        textbox_layout = QtWidgets.QVBoxLayout()
        textbox_layout.setContentsMargins(0, 0, 0, 0)
        textbox_layout.setSpacing(6)

        buttons_layout = QtWidgets.QVBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        textbox_layout.addWidget(QtWidgets.QLabel("Текст:"))
        owner._chat_text_edit = QtWidgets.QTextEdit()
        owner._chat_text_edit.setPlaceholderText("Введите сообщение...")
        owner._chat_text_edit.setMinimumHeight(120)
        owner._chat_text_edit.setMaximumHeight(180)
        owner._chat_text_edit.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        owner._chat_text_edit.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        owner._chat_text_edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        textbox_layout.addWidget(owner._chat_text_edit)

        buttons_layout.addWidget(QtWidgets.QLabel("Вложения:"))
        attach_row = QtWidgets.QVBoxLayout()
        attach_row.setContentsMargins(0, 0, 0, 0)
        attach_row.setSpacing(8)
        owner._chat_add_files_btn = QtWidgets.QPushButton("Добавить файлы")
        owner._chat_add_files_btn.clicked.connect(owner._chat_pick_files)
        owner._chat_add_files_btn.setMinimumHeight(30)
        attach_row.addWidget(owner._chat_add_files_btn)
        owner._chat_drop_area = QtWidgets.QFrame()
        owner._chat_drop_area.setFrameShape(QtWidgets.QFrame.Shape.Box)
        owner._chat_drop_area.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        owner._chat_drop_area.setLineWidth(1)
        owner._chat_drop_area.setAcceptDrops(True)
        owner._chat_drop_area.setToolTip("Перетащите файлы сюда (до 20MB каждый, до 5 штук)")
        owner._chat_drop_area.setMinimumHeight(56)
        owner._chat_drop_area.setMaximumHeight(72)
        drop_layout = QtWidgets.QVBoxLayout()
        drop_layout.setContentsMargins(8, 2, 8, 2)
        owner._chat_drop_label = QtWidgets.QLabel("Drag & Drop файлов/фото сюда")
        owner._chat_drop_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(owner._chat_drop_label)
        owner._chat_drop_area.setLayout(drop_layout)
        owner._chat_drop_area.installEventFilter(owner)
        attach_row.addWidget(owner._chat_drop_area, 0)
        buttons_layout.addLayout(attach_row)
        buttons_layout.addStretch()

        owner._chat_selected_files_label = QtWidgets.QLabel("Выбранные файлы:")
        send_box_layout.addWidget(owner._chat_selected_files_label)
        owner._chat_selected_files_list = QtWidgets.QListWidget()
        owner._chat_selected_files_list.setMaximumHeight(90)
        owner._chat_selected_files_label.setVisible(False)
        owner._chat_selected_files_list.setVisible(False)
        send_box_layout.addWidget(owner._chat_selected_files_list)

        send_btn_row = QtWidgets.QHBoxLayout()
        send_btn_row.addStretch()
        owner._chat_send_btn = QtWidgets.QPushButton("Отправить")
        owner._chat_send_btn.setMinimumWidth(120)
        owner._chat_send_btn.clicked.connect(owner._chat_send_message)
        send_btn_row.addWidget(owner._chat_send_btn)
        buttons_layout.addLayout(send_btn_row)
        owner._chat_send_status = QtWidgets.QLabel("")
        owner._chat_send_status.setWordWrap(True)
        send_box_layout.addWidget(owner._chat_send_status)

        send_layout.addLayout(textbox_layout)
        send_layout.addLayout(buttons_layout)
        send_layout.setStretch(0, 4)
        send_layout.setStretch(1, 2)

        send_box_layout.addLayout(send_layout)

        send_grp.setLayout(send_box_layout)
        send_grp.setMinimumHeight(240)
        chat_tab_layout.addWidget(send_grp, 1)
        chat_tab.setLayout(chat_tab_layout)

        return chat_tab
