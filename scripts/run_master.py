from __future__ import annotations

import argparse
import logging
import os
import threading
from concurrent import futures
from pathlib import Path

import grpc

from cluster_core.common.config import load_master_config, WorkerConfig
from cluster_core.common.log_buffer import LogBuffer, make_buffer_handler
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

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_buffer = LogBuffer(max_lines=2000)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(make_buffer_handler(log_buffer, fmt))
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger("master")

    # Если мастер запущен из GUI, даём приоритет настройкам автосейва (workers, режим загрузки, ресурсы).
    if os.environ.get("CHABOSS_CLUSTER_FROM_GUI") == "1":
        try:
            from ui.settings_store import load as load_gui_settings
        except Exception:
            load_gui_settings = None

        if load_gui_settings is not None:
            gui_settings = load_gui_settings()
            workers_cfg = [
                WorkerConfig(
                    host=w["host"],
                    port=int(w["port"]),
                    auth_token=w.get("auth_token") or None,
                )
                for w in gui_settings.get("workers", [])
                if w.get("host") and int(w.get("port") or 0) > 0
            ]
            model_load_mode = gui_settings.get("model_load_mode", cfg.model_load_mode)
            resource_usage_percent = int(gui_settings.get("resource_usage_percent", cfg.resource_usage_percent))
            resource_usage_percent = max(1, min(100, resource_usage_percent))

            if workers_cfg:
                cfg.workers = workers_cfg
            cfg.model_load_mode = model_load_mode
            cfg.resource_usage_percent = resource_usage_percent

            logger.info(
                "Master started from GUI: applying GUI settings "
                "(workers=%d, model_load_mode=%s, resource_usage_percent=%d)",
                len(cfg.workers),
                cfg.model_load_mode,
                cfg.resource_usage_percent,
            )

    logger.info("Config loaded from %s (%d workers)", config_path, len(cfg.workers))

    registry = WorkerRegistry()
    master_node = MasterNode(cfg, registry, log_buffer=log_buffer)

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

