from __future__ import annotations

import argparse
import logging
import threading
from concurrent import futures
from pathlib import Path

import grpc

from cluster_core.common.config import load_master_config
from cluster_core.master.worker_registry import WorkerRegistry
from cluster_core.master.master_node import MasterNode
from cluster_core.master.admin_service import MasterAdminService
from cluster_core.grpc import cluster_pb2_grpc
from cluster_core.api.openai_http import create_app


def _default_config_path() -> str:
    """Путь к config/master.yaml относительно корня проекта (не от cwd)."""
    root = Path(__file__).resolve().parent.parent
    return str(root / "config" / "master.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run master node.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to master configuration file (default: <project>/config/master.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config or _default_config_path())
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    cfg = load_master_config(config_path)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("master")
    logger.info("Config loaded from %s (%d workers)", config_path, len(cfg.workers))

    registry = WorkerRegistry()
    master_node = MasterNode(cfg, registry)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    cluster_pb2_grpc.add_MasterAdminServiceServicer_to_server(
        MasterAdminService(registry, master_node), server
    )

    listen_addr = f"{cfg.listen_host}:{cfg.listen_port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting master on %s", listen_addr)
    server.start()

    # OpenAI-совместимый HTTP API в фоновом потоке
    http_app = create_app(registry, master_node, cfg)

    def run_http() -> None:
        import uvicorn
        uvicorn.run(
            http_app,
            host=cfg.http_listen_host,
            port=cfg.http_listen_port,
            log_level="info",
        )

    http_thread = threading.Thread(target=run_http, daemon=True)
    http_thread.start()
    logger.info("OpenAI HTTP API on http://%s:%s", cfg.http_listen_host, cfg.http_listen_port)

    # Стартуем управление воркерами в фоне.
    master_node.start_background()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down master...")
        master_node.stop()
        server.stop(grace=None)


if __name__ == "__main__":
    main()

