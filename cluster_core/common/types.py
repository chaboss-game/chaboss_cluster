from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class WorkerStatus(Enum):
    ONLINE = auto()
    OFFLINE = auto()
    UNSTABLE = auto()
    RECONNECTING = auto()


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_vram_mb: int
    compute_capability: Optional[str] = None
    backend: str = "cuda"  # "cuda", "rocm", "directml", "metal", "none"


@dataclass
class ResourceInfo:
    cpu_cores: int
    ram_total_mb: int
    ram_available_mb: int
    gpus: List[GpuInfo] = field(default_factory=list)
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None


@dataclass(frozen=True)
class WorkerId:
    host: str
    port: int

    def as_str(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class WorkerDescriptor:
    worker_id: WorkerId
    status: WorkerStatus
    resources: ResourceInfo
    labels: Dict[str, str] = field(default_factory=dict)

