import asyncio
import threading
import queue
import cv2
import socketio
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceCandidate, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

class AnnotatedFrameTrack(VideoStreamTrack):
    kind = 'video'

    def __init__(self, annotated_q: queue.Queue):
        super().__init__()
        self.annotated_q = annotated_q
        self._last_frame = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame_bgr = None
        try:
            frame_bgr = self.annotated_q.get(timeout=0.1)
        except queue.Empty:
            frame_bgr = self._last_frame

        if frame_bgr is None:
            import numpy as np
            frame_bgr = np.zeros((480, 640, 3), dtype='uint8')

        self._last_frame = frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class WebRTCPublisherThread(threading.Thread):
    def __init__(self, annotated_q: queue.Queue, signaling_url: str):
        super().__init__(daemon=True)
        self.annotated_q   = annotated_q
        self.signaling_url = signaling_url
        self._stop_event   = threading.Event()
        self.loop          = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        finally:
            self.loop.close()

    def stop(self):
        self._stop_event.set()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

    async def _main(self):
        NS = '/signaling'
        sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=10)
        peer_connections: dict[str, RTCPeerConnection] = {}

        async def make_pc(viewer_id: str) -> RTCPeerConnection:
            # ✅ Fix 1 — dùng RTCConfiguration thay vì dict
            pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            peer_connections[viewer_id] = pc
            pc.addTrack(AnnotatedFrameTrack(self.annotated_q))

            @pc.on('icecandidate')
            async def on_ice(candidate):
                if candidate:
                    await sio.emit('ice-candidate', {
                        'candidate': {
                            'candidate':     candidate.candidate,
                            'sdpMid':        candidate.sdpMid,
                            'sdpMLineIndex': candidate.sdpMLineIndex,
                        },
                        'targetId': viewer_id,
                    }, namespace=NS)

            @pc.on('connectionstatechange')
            async def on_state():
                state = pc.connectionState
                print(f'[WebRTC] Viewer {viewer_id}: {state}')
                if state in ('failed', 'closed', 'disconnected'):
                    peer_connections.pop(viewer_id, None)
                    await pc.close()

            return pc

        @sio.on('connect', namespace=NS)
        async def on_connect():
            print('[WebRTC] Signaling server connected')
            await sio.emit('register-publisher', namespace=NS)

        @sio.on('connect_error', namespace=NS)
        async def on_connect_error(data):
            print(f'[WebRTC] Connection error: {data}')

        @sio.on('disconnect', namespace=NS)
        async def on_disconnect():
            print('[WebRTC] Signaling server disconnected')

        @sio.on('offer', namespace=NS)
        async def on_offer(data):
            viewer_id = data['viewerId']
            print(f'[WebRTC] Offer từ viewer: {viewer_id}')
            pc = await make_pc(viewer_id)

            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=data['sdp']['sdp'],
                type=data['sdp']['type'],
            ))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            await sio.emit('answer', {
                'sdp': {
                    'sdp':  pc.localDescription.sdp,
                    'type': pc.localDescription.type,
                },
                'viewerId': viewer_id,
            }, namespace=NS)

        @sio.on('ice-candidate', namespace=NS)
        async def on_ice(data):
            viewer_id = data.get('fromId')
            pc = peer_connections.get(viewer_id)
            if not pc:
                return
            c = data['candidate']
            parts = c['candidate'].split()
            try:
                # ✅ Fix 2 — dùng 'ip' thay vì 'host'
                candidate = RTCIceCandidate(
                    foundation=parts[0].replace('candidate:', ''),
                    component=int(parts[1]),
                    protocol=parts[2],
                    priority=int(parts[3]),
                    ip=parts[4],
                    port=int(parts[5]),
                    type=parts[7],
                    sdpMid=c.get('sdpMid'),
                    sdpMLineIndex=c.get('sdpMLineIndex'),
                )
                await pc.addIceCandidate(candidate)
            except Exception as e:
                print(f'[WebRTC] ICE parse error: {e}')

        await sio.connect(
            self.signaling_url,
            namespaces=[NS],
            socketio_path='/socket.io',
        )

        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)

        for pc in list(peer_connections.values()):
            await pc.close()
        await sio.disconnect()