from __future__ import annotations

import argparse
import logging
from concurrent import futures

import grpc

from cluster_core.common.config import load_worker_config
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

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("worker")

    host = cfg.get("listen_host", "0.0.0.0")
    port = int(cfg.get("listen_port", 50052))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))

    worker_service = WorkerService(host=host, port=port)
    from cluster_core.grpc import cluster_pb2_grpc

    cluster_pb2_grpc.WorkerServiceServicer_to_server(worker_service, server)

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
