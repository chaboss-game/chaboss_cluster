"""UI tab widgets for MainWindow."""

from .chat_tab import ChatTabWidget
from .cluster_tab import ClusterTabWidget
from .settings_tab import SettingsTabWidget
from .worker_tab import WorkerTabWidget

__all__ = [
    "ClusterTabWidget",
    "SettingsTabWidget",
    "ChatTabWidget",
    "WorkerTabWidget",
]
