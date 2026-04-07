import cv2
import time

def calc_fps(prev_time: float) -> float:
    """Tính FPS dựa trên thời điểm frame trước."""
    now = time.time()
    return 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0


def draw_detections(frame, detections: list, fps: float):
    """
    Vẽ bounding box, label, FPS và drone count lên frame.
    Trả về frame đã annotate (không modify in-place).
    """
    annotated = frame.copy()
    drone_count = len(detections)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        area = det["area"]

        thickness = 1 if area < 2000 else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        label = f"D {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # FPS
    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Drone count
    cv2.putText(annotated, f"Drones: {drone_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Cảnh báo
    if drone_count > 0:
        cv2.putText(annotated, "DRONE DETECTED!", (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return annotated


def area_filter(box, conf: float, area_thresh: int,
                conf_small: float, conf_large: float) -> bool:
    """
    Trả về True nếu detection vượt ngưỡng conf theo diện tích.
    Dùng lại ở InferenceThread._filter nếu muốn tách logic ra đây.
    """
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    thresh = conf_small if area < area_thresh else conf_large
    return conf >= thresh