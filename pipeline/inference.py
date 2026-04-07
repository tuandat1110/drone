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
        while not self.stop_event.is_set():
            frame = self.frame_queue.get()  # Block until a frame is available
            if frame is None:
                break
            h, w = frame.shape[:2]
            scale = cfg.IMGSZ / max(h, w)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

            result = self.model(resized, conf=cfg.CONF_THRESHOLD, imgsz=cfg.IMGSZ, verbose=False)
            detection = self._filter(result[0].boxes)
            self.result_queue.put((frame, detection))

    def _filter(self, boxes):
        dets = []
        for box in boxes:
            if int(box.cls[0]) != cfg.DRONE_CLS: continue
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)
            thresh = cfg.CONF_SMALL if area < cfg.AREA_THRESH else cfg.CONF_LARGE
            if conf >= thresh:
                dets.append({"box": (x1,y1,x2,y2), "conf": conf, "area": area})
        return dets

    def stop(self):
        self.stop_event.set()
        self.frame_queue.put(None)