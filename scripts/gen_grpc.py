#!/usr/bin/env python3
"""
Генерация Python-модулей из proto/cluster.proto.
Запуск из корня проекта: pip install -r requirements.txt && python scripts/gen_grpc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = PROJECT_ROOT / "proto"
OUT_DIR = PROJECT_ROOT / "cluster_core" / "grpc"
PROTO_FILE = PROTO_DIR / "cluster.proto"


def main() -> int:
    try:
        import grpc_tools.protoc as protoc_main
    except ImportError:
        print("Установите зависимости: pip install -r requirements.txt", file=sys.stderr)
        return 1

    out_dir = str(OUT_DIR)
    proto_include = str(PROTO_DIR)
    proto_file = str(PROTO_FILE)

    argv = [
        "grpc_tools.protoc",
        f"-I{proto_include}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        proto_file,
    ]
    status = protoc_main.main(argv)
    if status != 0:
        return status

    # Исправление импорта в _grpc: иначе "import cluster_pb2" не находит модуль внутри пакета.
    grpc_file = OUT_DIR / "cluster_pb2_grpc.py"
    if grpc_file.exists():
        text = grpc_file.read_text(encoding="utf-8")
        if "import cluster_pb2" in text and "from . import cluster_pb2" not in text:
            text = text.replace("import cluster_pb2 as ", "from . import cluster_pb2 as ")
            grpc_file.write_text(text, encoding="utf-8")

    print("Generated:", OUT_DIR / "cluster_pb2.py", OUT_DIR / "cluster_pb2_grpc.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
