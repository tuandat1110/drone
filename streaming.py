from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time
import queue
from threading import Thread

app = Flask(__name__)

# ===== Model =====
model = YOLO('models/best_7.pt')
DRONE_CLASS_ID = 0

# ===== Video =====
video_path = "./video/drone2.mp4"
cap = cv2.VideoCapture(video_path)

# ===== Queues =====
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

# Biến để kiểm soát trạng thái dừng
running = True

def capture_frames():
    global running
    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(frame)
    cap.release()

def detect_frames():
    global running
    prev_time = time.time()
    
    while running or not frame_queue.empty():
        try:
            # Timeout để không bị kẹt cứng khi hết video
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Resize tối ưu cho YOLO
        h, w, _ = frame.shape
        scale = 640 / max(h, w)
        frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Inference (chỉ định device='cpu' hoặc '0' nếu có GPU)
        results = model(frame_resized, conf=0.25, imgsz=640, verbose=False)
        r = results[0]
        
        annotated_frame = frame_resized.copy()
        drone_count = 0

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != DRONE_CLASS_ID: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            # Logic lọc threshold của bạn
            if area < 1500 and conf < 0.3: continue
            if area >= 1500 and conf < 0.5: continue

            drone_count += 1
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"D {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Drones: {drone_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty: pass
        result_queue.put(annotated_frame)

def generate():
    print("-> Đã bắt đầu luồng stream cho trình duyệt")
    while True:
        # Lấy frame từ result_queue để stream
        frame = result_queue.get()
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    Thread(target=capture_frames, daemon=True).start()
    Thread(target=detect_frames, daemon=True).start()
    # Debug mode nên tắt khi chạy production để tránh lỗi thread
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)