import cv2
from ultralytics import YOLO
import time

model = YOLO('models/best_7.pt')

cap = cv2.VideoCapture("./video/drone3.mp4")
#cap = cv2.VideoCapture(0)

prev_time = time.time()

DRONE_CLASS_ID = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ===== 1. KHÔNG resize vuông cứng (giữ tỉ lệ) =====
    h, w, _ = frame.shape
    scale = 960 / max(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # ===== 2. INFERENCE với imgsz lớn hơn =====
    results = model(frame, conf=0.25, imgsz=960)  # giảm conf để bắt drone nhỏ
    r = results[0]

    annotated_frame = frame.copy()
    drone_count = 0

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != DRONE_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        # ===== 3. THRESHOLD ĐỘNG (QUAN TRỌNG) =====
        if area < 1500:
            if conf < 0.3:
                continue
        else:
            if conf < 0.5:
                continue

        drone_count += 1

        # ===== 4. VẼ BOX RÕ HƠN CHO OBJECT NHỎ =====
        thickness = 1 if area < 2000 else 2
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), thickness)

        label = f"D {conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ===== 5. FPS =====
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Drones: {drone_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if drone_count > 0:
        cv2.putText(annotated_frame, "DRONE DETECTED!", (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Drone Detection SMALL OPTIMIZED", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
