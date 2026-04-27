import cv2
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = YOLO("models/best_7.pt")
relay = MediaRelay()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        # Nhận frame từ track gốc (camera)
        frame = await self.track.recv()
        
        # Chuyển đổi frame sang mảng numpy (OpenCV format)
        img = frame.to_ndarray(format="bgr24")

        # --- XỬ LÝ AI YOLOv8 ---
        results = model.predict(img, conf=0.5, verbose=False)
        annotated_img = results[0].plot()
        # -----------------------

        # Chuyển ngược lại về định dạng frame của WebRTC
        new_frame = VideoFrame.from_ndarray(annotated_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()

    # Mở Camera bằng OpenCV thông qua aiortc media relay hoặc trực tiếp
    # Ở đây dùng camera mặc định
    cap = cv2.VideoCapture(0)
    
    # Một helper đơn giản để đọc camera cho aiortc
    class CameraStream(VideoStreamTrack):
        def __init__(self):
            super().__init__()
            self.cap = cap
        async def recv(self):
            pts, time_base = await self.next_timestamp()
            ret, frame = self.cap.read()
            if not ret: return
            new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            new_frame.pts = pts
            new_frame.time_base = time_base
            return new_frame

    video_track = CameraStream()
    # Gắn AI Transform vào luồng camera
    pc.addTrack(VideoTransformTrack(relay.subscribe(video_track)))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)