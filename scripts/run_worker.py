from __future__ import annotations

import argparse
import logging
import sys
from concurrent import futures
from pathlib import Path

import grpc

from cluster_core.common.config import load_worker_config
from cluster_core.common.log_buffer import LogBuffer, make_buffer_handler
from cluster_core.worker.worker_service import WorkerService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run worker node.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/worker.yaml",
        help="Path to worker configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_worker_config(args.config)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    log_buffer = LogBuffer(max_lines=2000)
    root.addHandler(make_buffer_handler(log_buffer, fmt))

    try:
        project_root = Path(__file__).resolve().parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "worker.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
    except Exception:  # права, путь, блокировка файла — воркер не должен падать
        pass
    logger = logging.getLogger("worker")
    if any(h for h in root.handlers if isinstance(h, logging.FileHandler)):
        logger.info("Логи воркера также пишутся в logs/worker.log (при запуске из GUI stderr часто отключён).")
    try:
        if sys.stderr is not None and not getattr(sys.stderr, "closed", False):
            sh = logging.StreamHandler(sys.stderr)
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter(fmt))
            root.addHandler(sh)
    except OSError:
        pass
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=fmt)

    host = cfg.get("listen_host", "0.0.0.0")
    port = int(cfg.get("listen_port", 50052))
    auth_token = cfg.get("auth_token") or None

    # Долгоживущие соединения: не закрывать по возрасту/простою (INT_MAX = без лимита).
    # Иначе сервер gRPC по умолчанию может закрывать соединение через несколько минут.
    GRPC_UNLIMITED_MS = 2147483647
    server_options = [
        ("grpc.keepalive_time_ms", 15_000),
        ("grpc.keepalive_timeout_ms", 5_000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.max_connection_idle_ms", GRPC_UNLIMITED_MS),
        ("grpc.max_connection_age_ms", GRPC_UNLIMITED_MS),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=server_options,
    )

    worker_service = WorkerService(host=host, port=port, auth_token=auth_token, log_buffer=log_buffer)
    from cluster_core.grpc import cluster_pb2_grpc

    cluster_pb2_grpc.add_WorkerServiceServicer_to_server(worker_service, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting worker on %s", listen_addr)
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
        server.stop(grace=None)


if __name__ == "__main__":
    main()
