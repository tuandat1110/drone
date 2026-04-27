# pipeline/display.py
import cv2, threading, time
from core.utils import draw_detections, calc_fps
from core.alert import AlertHandler

class DisplayProcessor: # Đổi tên thành Processor cho đúng bản chất
    def __init__(self):
        self.alert = AlertHandler()
        self._prev_time = time.time()

    def process_frame(self, item):
        if item is None: return None
        frame, dets = item

        fps = calc_fps(self._prev_time)
        self._prev_time = time.time()

        # Vẽ kết quả lên frame
        annotated = draw_detections(frame, dets, fps)

        # Gửi cảnh báo nếu có drone
        # if dets:
        #     self.alert.trigger(frame, dets)
        
        return annotated