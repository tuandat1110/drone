# pipeline/webrtc_publisher.py
import asyncio
import threading
import queue
import cv2
import socketio
import requests
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
    def __init__(self, annotated_q: queue.Queue, backend_url: str, device_key: str):
        super().__init__(daemon=True)
        self.annotated_q  = annotated_q
        self.backend_url  = backend_url   # 'http://localhost:3000'
        self.device_key   = device_key    # đọc từ .env
        self._stop_event  = threading.Event()
        self.loop         = None

    def _get_session_token(self) -> str:
        """Bước 1: đổi device_key lấy session_token"""
        res = requests.post(
            f'{self.backend_url}/devices/session',
            json={'device_key': self.device_key},
            timeout=10,
        )
        res.raise_for_status()  
        body = res.json()
        token = body.get("data", {}).get("session_token")
        if not token:
            raise ValueError("Backend không trả về session_token")

        print(f'[WebRTC] Session token: {token[:8]}...')
        return token

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
        # Bước 1: lấy session token trước khi connect WebSocket
        try:
            session_token = self._get_session_token()
        except Exception as e:
            print(f'[WebRTC] Không thể lấy session token: {e}')
            return

        NS = '/signaling'
        sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=10)
        peer_connections: dict[str, RTCPeerConnection] = {}

        async def make_pc(viewer_id: str) -> RTCPeerConnection:
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
            # Bước 2: sau khi kết nối WS, KHÔNG emit register-publisher nữa
            # handleConnection bên NestJS đã tự xử lý qua session_token
            print('[WebRTC] Signaling connected — authenticated as device')

        @sio.on('connect_error', namespace=NS)
        async def on_connect_error(data):
            print(f'[WebRTC] Connection error: {data}')

        @sio.on('disconnect', namespace=NS)
        async def on_disconnect():
            print('[WebRTC] Signaling disconnected')

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

        # Bước 2: kết nối WS với session_token trong auth
        await sio.connect(
            self.backend_url,
            namespaces=[NS],
            socketio_path='/socket.io',
            auth={
                'token': session_token,
                'type': 'device',       # NestJS dùng type này để phân nhánh
            },
        )

        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)

        for pc in list(peer_connections.values()):
            await pc.close()
        await sio.disconnect()