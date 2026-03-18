"""
Admin gRPC-сервис мастера для UI и внешних инструментов (ListWorkers, LoadModel, LoadModelStream).
"""
from __future__ import annotations

import queue
import threading

import grpc

from cluster_core.common.types import WorkerDescriptor, WorkerStatus
from cluster_core.master.worker_registry import WorkerRegistry
from cluster_core.grpc import cluster_pb2, cluster_pb2_grpc
from cluster_core.common.chat_storage import StoredChatAttachment
import base64

_STATUS_TO_PROTO = {
    WorkerStatus.ONLINE: cluster_pb2.WORKER_STATUS_ONLINE,
    WorkerStatus.OFFLINE: cluster_pb2.WORKER_STATUS_OFFLINE,
    WorkerStatus.UNSTABLE: cluster_pb2.WORKER_STATUS_UNSTABLE,
    WorkerStatus.RECONNECTING: cluster_pb2.WORKER_STATUS_RECONNECTING,
}


def _worker_descriptor_to_proto(wd: WorkerDescriptor) -> cluster_pb2.WorkerDescriptor:
    gpus = [
        cluster_pb2.GpuInfo(
            index=g.index,
            name=g.name,
            total_vram_mb=g.total_vram_mb,
            compute_capability=g.compute_capability or "",
            backend=g.backend,
        )
        for g in wd.resources.gpus
    ]
    resources = cluster_pb2.ResourceInfo(
        cpu_cores=wd.resources.cpu_cores,
        ram_total_mb=wd.resources.ram_total_mb,
        ram_available_mb=wd.resources.ram_available_mb,
        gpus=gpus,
        torch_version=wd.resources.torch_version or "",
        cuda_version=wd.resources.cuda_version or "",
        rocm_version=wd.resources.rocm_version or "",
        os_name=wd.resources.os_name or "",
        os_version=wd.resources.os_version or "",
    )
    return cluster_pb2.WorkerDescriptor(
        id=cluster_pb2.WorkerId(host=wd.worker_id.host, port=wd.worker_id.port),
        status=_STATUS_TO_PROTO.get(wd.status, cluster_pb2.WORKER_STATUS_OFFLINE),
        resources=resources,
        token_fingerprint=wd.token_fingerprint or "",
        token_status=wd.token_status or "UNKNOWN",
    )


class MasterAdminService(cluster_pb2_grpc.MasterAdminServiceServicer):
    """Сервис админ-API мастера: список воркеров, загрузка модели."""

    def __init__(self, registry: WorkerRegistry, master_node: object = None) -> None:
        self._registry = registry
        self._master_node = master_node

    def ListWorkers(
        self,
        request: cluster_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.WorkerList:
        workers = [
            _worker_descriptor_to_proto(wd)
            for wd in self._registry.all().values()
        ]
        return cluster_pb2.WorkerList(workers=workers)

    def LoadModel(
        self,
        request: cluster_pb2.LoadModelRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.LoadModelResponse:
        if not request.hf_model_id or not request.hf_model_id.strip():
            return cluster_pb2.LoadModelResponse(ok=False, error="Укажите hf_model_id")
        if self._master_node is None:
            return cluster_pb2.LoadModelResponse(ok=False, error="Мастер не готов к загрузке модели")
        ok, err = self._master_node.load_model(request.hf_model_id.strip())
        return cluster_pb2.LoadModelResponse(ok=ok, error=err or "")

    def LoadModelStream(
        self,
        request: cluster_pb2.LoadModelRequest,
        context: grpc.ServicerContext,
    ):
        if not request.hf_model_id or not request.hf_model_id.strip():
            yield cluster_pb2.LoadModelProgressEvent(done=True, ok=False, error="Укажите hf_model_id")
            return
        if self._master_node is None:
            yield cluster_pb2.LoadModelProgressEvent(done=True, ok=False, error="Мастер не готов к загрузке модели")
            return
        progress_queue: queue.Queue[cluster_pb2.LoadModelProgressEvent] = queue.Queue()
        def run() -> None:
            self._master_node.load_model_with_progress(request.hf_model_id.strip(), progress_queue)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while True:
            try:
                ev = progress_queue.get(timeout=0.5)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
            yield ev
            if ev.done:
                break

    def UnloadModel(
        self,
        request: cluster_pb2.UnloadModelRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.UnloadModelResponse:
        if self._master_node is None:
            return cluster_pb2.UnloadModelResponse(ok=False, error="Мастер не готов")
        model_id = (request.model_id or "").strip() or None
        ok, err = self._master_node.unload_model(model_id)
        return cluster_pb2.UnloadModelResponse(ok=ok, error=err or "")

    def UpdateWorkersConfig(
        self,
        request: cluster_pb2.UpdateWorkersConfigRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.UpdateWorkersConfigResponse:
        if self._master_node is None:
            return cluster_pb2.UpdateWorkersConfigResponse(ok=False, error="Мастер не готов")
        from cluster_core.common.config import WorkerConfig
        workers = [
            WorkerConfig(host=w.host or "", port=int(w.port) if w.port else 0, auth_token=w.auth_token or None)
            for w in request.workers
        ]
        workers = [w for w in workers if w.host and w.port > 0]
        try:
            ok, err = self._master_node.update_workers_config(workers)
            return cluster_pb2.UpdateWorkersConfigResponse(ok=ok, error=err or "")
        except Exception as e:
            return cluster_pb2.UpdateWorkersConfigResponse(ok=False, error=str(e))

    def RemoteUpdateWorkers(
        self,
        request: cluster_pb2.RemoteUpdateWorkersRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.RemoteUpdateWorkersResponse:
        if self._master_node is None:
            return cluster_pb2.RemoteUpdateWorkersResponse(ok=False, error="Мастер не готов")
        ok, err, results = self._master_node.remote_update_workers(
            restart_gui=bool(request.restart_gui),
            start_worker=bool(request.start_worker),
            git_remote=(request.git_remote or "origin").strip() or "origin",
            git_branch=(request.git_branch or "").strip(),
        )
        resp = cluster_pb2.RemoteUpdateWorkersResponse(ok=ok, error=err or "")
        for r in results:
            resp.results.append(
                cluster_pb2.RemoteWorkerUpdateResult(
                    worker=r.get("worker", ""),
                    ok=bool(r.get("ok", False)),
                    error=r.get("error", "") or "",
                    output=r.get("output", "") or "",
                )
            )
        return resp

    def ListChatChannels(
        self,
        request: cluster_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.ChatChannelsResponse:
        channels = self._master_node._chat_storage.list_channels()
        return cluster_pb2.ChatChannelsResponse(
            channels=[
                cluster_pb2.ChatChannelInfo(id=cid, name=name)
                for cid, name in channels
            ]
        )

    def MutateChatChannels(
        self,
        request: cluster_pb2.ChatChannelsMutationRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.ChatChannelsResponse:
        ops: list[dict] = []
        for op in request.ops:
            t = op.type
            if t == cluster_pb2.ChatChannelMutationType.CHAT_CHANNEL_CREATE:
                ops.append({"type": "create", "channel_id": op.channel_id, "name": op.name})
            elif t == cluster_pb2.ChatChannelMutationType.CHAT_CHANNEL_RENAME:
                ops.append({"type": "rename", "channel_id": op.channel_id, "name": op.name})
            elif t == cluster_pb2.ChatChannelMutationType.CHAT_CHANNEL_DELETE:
                ops.append({"type": "delete", "channel_id": op.channel_id, "name": op.name})
            else:
                return cluster_pb2.ChatChannelsResponse(channels=[])
        self._master_node._chat_storage.mutate_channels(ops)
        channels = self._master_node._chat_storage.list_channels()
        return cluster_pb2.ChatChannelsResponse(
            channels=[cluster_pb2.ChatChannelInfo(id=cid, name=name) for cid, name in channels]
        )

    def GetChatHistory(
        self,
        request: cluster_pb2.GetChatHistoryRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.ChatHistoryResponse:
        channel_id = (request.channel_id or "").strip()
        since_seq = int(request.since_seq or 0)
        limit = int(request.limit or 50)
        if not channel_id:
            return cluster_pb2.ChatHistoryResponse(messages=[], next_seq=0)

        msgs, next_seq = self._master_node._chat_storage.get_history(
            channel_id=channel_id, since_seq=since_seq, limit=limit
        )

        out_msgs: list[cluster_pb2.ChatMessage] = []
        for m in msgs:
            attachments_proto: list[cluster_pb2.ChatAttachmentMeta] = []
            for a in m.get("attachments") or []:
                thumb_b64 = a.get("thumbnail_jpeg_b64")
                thumb_bytes = base64.b64decode(thumb_b64.encode("ascii")) if thumb_b64 else b""
                attachments_proto.append(
                    cluster_pb2.ChatAttachmentMeta(
                        attachment_id=str(a.get("attachment_id") or ""),
                        filename=str(a.get("filename") or ""),
                        mime_type=str(a.get("mime_type") or ""),
                        is_image=bool(a.get("is_image") or False),
                        size=int(a.get("size") or 0),
                        thumbnail_jpeg=thumb_bytes,
                    )
                )
            out_msgs.append(
                cluster_pb2.ChatMessage(
                    message_id=str(m.get("message_id") or ""),
                    seq=int(m.get("seq") or 0),
                    timestamp_ms=int(m.get("timestamp_ms") or 0),
                    channel_id=str(m.get("channel_id") or ""),
                    sender=str(m.get("sender") or ""),
                    text=str(m.get("text") or ""),
                    attachments=attachments_proto,
                )
            )

        return cluster_pb2.ChatHistoryResponse(messages=out_msgs, next_seq=int(next_seq or 0))

    def GetChatAttachment(
        self,
        request: cluster_pb2.GetChatAttachmentRequest,
        context: grpc.ServicerContext,
    ):
        message_id = str(request.message_id or "").strip()
        attachment_id = str(request.attachment_id or "").strip()
        if not attachment_id:
            return

        try:
            path = self._master_node._chat_storage.attachment_original_path(attachment_id)
        except Exception:
            return

        CHUNK = 256 * 1024
        with path.open("rb") as f:
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                # is_last: узнаём EOF через peek на 1 байт
                pos = f.tell()
                b2 = f.read(1)
                is_last = b2 == b""
                if not is_last:
                    f.seek(pos)
                yield cluster_pb2.ChatAttachmentChunk(data=b, is_last=is_last)

    def PostChatMessage(
        self,
        request_iterator,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.PostChatMessageResponse:
        """
        Client-streaming: сначала header, затем чанки для каждого attachment_id.
        """
        header = None
        header_attachments = []
        # tmp file handles
        tmp_fds: dict[str, Any] = {}
        bytes_received: dict[str, int] = {}
        is_last_seen: dict[str, bool] = {}
        try:
            for chunk in request_iterator:
                which = chunk.WhichOneof("payload")
                if which == "header":
                    header = chunk.header
                    header_attachments = list(header.attachments or [])
                    if len(header_attachments) > 5:
                        return cluster_pb2.PostChatMessageResponse(ok=False, error="Слишком много вложений (макс 5).", message_id=header.message_id)
                    if not header.channel_id:
                        return cluster_pb2.PostChatMessageResponse(ok=False, error="channel_id обязателен.", message_id=header.message_id)
                    # открываем tmp-файлы под каждое вложение
                    for a in header_attachments:
                        if a.size > 20 * 1024 * 1024:
                            return cluster_pb2.PostChatMessageResponse(ok=False, error=f"Слишком большой файл (>{20}MB): {a.filename}", message_id=header.message_id)
                        path = self._master_node._chat_storage.attachment_original_path(a.attachment_id)
                        tmp = path.with_suffix(".tmp")
                        tmp.parent.mkdir(parents=True, exist_ok=True)
                        tmp_fds[a.attachment_id] = tmp.open("wb")
                        bytes_received[a.attachment_id] = 0
                        is_last_seen[a.attachment_id] = False
                elif which == "attachment_chunk":
                    if header is None:
                        return cluster_pb2.PostChatMessageResponse(ok=False, error="Сначала должен прийти header.", message_id="")
                    ac = chunk.attachment_chunk
                    aid = str(ac.attachment_id or "").strip()
                    if not aid:
                        continue
                    if aid not in tmp_fds:
                        # пропускаем неизвестные aid
                        continue
                    data = ac.data or b""
                    tmp_fds[aid].write(data)
                    bytes_received[aid] += len(data)
                    is_last_seen[aid] = bool(is_last_seen.get(aid, False) or ac.is_last)
                    if header_attachments:
                        # лимит по размеру
                        for a in header_attachments:
                            if a.attachment_id == aid:
                                if bytes_received[aid] > int(a.size or 0):
                                    return cluster_pb2.PostChatMessageResponse(ok=False, error="Превышен заявленный size вложения.", message_id=header.message_id)
                                break
                else:
                    continue
        except Exception as e:  # noqa: BLE001
            return cluster_pb2.PostChatMessageResponse(ok=False, error=str(e), message_id=header.message_id if header else "")
        finally:
            # На ошибки закрываем tmp-файлы, чтобы воркер/мастер не держал дескрипторы
            for fd in tmp_fds.values():
                try:
                    fd.flush()
                except Exception:
                    pass
                try:
                    fd.close()
                except Exception:
                    pass

        if header is None:
            return cluster_pb2.PostChatMessageResponse(ok=False, error="Header не получен.", message_id="")

        # финализируем tmp-файлы и сохраняем thumbnails/сообщение
        try:
            # close + replace
            for aid, fd in tmp_fds.items():
                fd.flush()
                fd.close()
                path = self._master_node._chat_storage.attachment_original_path(aid)
                tmp = path.with_suffix(".tmp")
                tmp.replace(path)

            # verify all last seen
            for a in header_attachments:
                if not is_last_seen.get(a.attachment_id, False):
                    return cluster_pb2.PostChatMessageResponse(ok=False, error=f"Не получены все чанки для вложения {a.filename}", message_id=header.message_id)

            # thumbnails
            attachments_for_record: list[StoredChatAttachment] = []
            for a in header_attachments:
                thumb_bytes = bytes(a.thumbnail_jpeg or b"")
                if a.is_image and thumb_bytes:
                    self._master_node._chat_storage.write_thumbnail_bytes(a.attachment_id, thumb_bytes)
                    thumb_b64 = base64.b64encode(thumb_bytes).decode("ascii")
                else:
                    thumb_b64 = None
                attachments_for_record.append(
                    StoredChatAttachment(
                        attachment_id=a.attachment_id,
                        filename=a.filename,
                        mime_type=a.mime_type,
                        is_image=bool(a.is_image),
                        size=int(a.size or 0),
                        thumbnail_jpeg_b64=thumb_b64,
                    )
                )

            seq = self._master_node._chat_storage.append_message(
                message_id=header.message_id,
                timestamp_ms=int(header.timestamp_ms or 0),
                channel_id=header.channel_id,
                sender=header.sender or "",
                text=header.text or "",
                attachments=attachments_for_record,
            )

            # Forward to selected workers (best effort, async)
            self._master_node.forward_chat_message_to_workers(header, header_attachments)
            return cluster_pb2.PostChatMessageResponse(ok=True, error="", message_id=header.message_id)
        except Exception as e:  # noqa: BLE001
            return cluster_pb2.PostChatMessageResponse(ok=False, error=str(e), message_id=header.message_id)

    def GetClusterLogs(
        self,
        request: cluster_pb2.GetClusterLogsRequest,
        context: grpc.ServicerContext,
    ) -> cluster_pb2.GetClusterLogsResponse:
        """Логи мастера и воркеров для отображения в GUI."""
        if self._master_node is None:
            return cluster_pb2.GetClusterLogsResponse(master_next=0, worker_logs={})
        worker_since = dict(request.worker_since) if request.worker_since else {}
        return self._master_node.get_cluster_logs(
            master_since=max(0, request.master_since or 0),
            worker_since=worker_since,
        )
