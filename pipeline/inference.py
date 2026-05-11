import cv2, threading
from ultralytics import YOLO
import config as cfg

class InferenceThread(threading.Thread):
    def __init__(self, frame_queue, result_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = threading.Event() 
        self.model = YOLO(cfg.MODEL_PATH)
    
    def run(self):
        print("Inference thread started.")
        while not self.stop_event.is_set():
            frame = self.frame_queue.get()
            if frame is None: break
            
            h_orig, w_orig, _ = frame.shape
            scale = cfg.IMGSZ / max(h_orig, w_orig)

            # 1. Resize để inference
            resized = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))

            # 2. Inference
            result = self.model(resized, conf=cfg.CONF_THRESHOLD, imgsz=cfg.IMGSZ)
            
            # 3. Truyền thêm scale vào hàm filter để tính toán lại tọa độ
            detection = self._filter(result[0].boxes, scale)
            
            # Trả về frame gốc và tọa độ đã được scale ngược lại
            self.result_queue.put((frame, detection))

    def _filter(self, boxes, scale):
        dets = []
        for box in boxes:
            if int(box.cls[0]) != cfg.DRONE_CLS: continue
            conf = float(box.conf[0])
            
            # Lấy tọa độ từ ảnh đã resize
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            
            # QUAN TRỌNG: Chia cho scale để đưa tọa độ về kích thước ảnh gốc
            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
            
            area = (x2 - x1) * (y2 - y1)
            thresh = cfg.CONF_SMALL if area < cfg.AREA_THRESH else cfg.CONF_LARGE
            if conf >= thresh:
                dets.append({"box": (x1, y1, x2, y2), "conf": conf, "area": area})
        return dets

    def stop(self):
        self.stop_event.set()
        self.frame_queue.put(None)