import cv2
from ultralytics import YOLO
import time
import queue
from threading import Thread

# ===== CẤU HÌNH =====
MODEL_PATH = 'models/best_7.pt'
VIDEO_SOURCE = 0 # Hoặc 0 nếu dùng Camera
DRONE_CLASS_ID = 0
IMGSZ = 640  # Kích thước ảnh đầu vào cho YOLO

# Khởi tạo Queue
# maxsize=1 là cực kỳ quan trọng cho Realtime để tránh bị trễ (Latency)
frame_queue = queue.Queue(maxsize=1) 
result_queue = queue.Queue(maxsize=1)

running = True

# 1. LUỒNG ĐỌC VIDEO (CAPTURE)
def capture_thread():
    global running
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    print(f"-> Đang kết nối nguồn: {VIDEO_SOURCE}")
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print("-> Hết video hoặc mất kết nối camera.")
            running = False
            break
        
        # Nếu queue đầy, xóa frame cũ để nạp frame mới nhất (Anti-lag)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except: pass
        frame_queue.put(frame)
        
    cap.release()

# 2. LUỒNG XỬ LÝ AI (INFERENCE)
def inference_thread():
    global running
    model = YOLO(MODEL_PATH)
    print("-> Model YOLO đã sẵn sàng.")
    
    while running:
        try:
            # Lấy frame từ hàng đợi (đợi tối đa 1s)
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Resize giữ nguyên tỉ lệ để không làm méo vật thể (Drone nhỏ dễ bị mất nếu méo)
        h, w = frame.shape[:2]
        scale = IMGSZ / max(h, w)
        frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Chạy AI
        results = model(frame_resized, conf=0.25, imgsz=IMGSZ, verbose=False)
        r = results[0]
        
        annotated_frame = frame_resized.copy()
        
        # Vẽ kết quả
        for box in r.boxes:
            if int(box.cls[0]) != DRONE_CLASS_ID: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Vẽ Box và Label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Drone {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Đẩy kết quả đã vẽ ra hàng đợi hiển thị
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except: pass
        result_queue.put(annotated_frame)

# 3. LUỒNG CHÍNH (DISPLAY)
if __name__ == '__main__':
    # Khởi chạy các luồng phụ
    t1 = Thread(target=capture_thread, daemon=True)
    t2 = Thread(target=inference_thread, daemon=True)
    t1.start()
    t2.start()

    prev_time = time.time()

    while running:
        try:
            # Lấy frame đã vẽ xong để hiện lên màn hình
            display_frame = result_queue.get(timeout=1)
            
            # Tính FPS hiển thị thực tế
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("DRONE DETECTION - DIRECT MONITOR", display_frame)
            
        except queue.Empty:
            continue

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()
    print("-> Hệ thống đã dừng.")