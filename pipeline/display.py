import cv2, threading, time
from core.utils import draw_detections, calc_fps
from core.alert import AlertHandler

class DisplayThread(threading.Thread):
    def __init__(self, result_queue):
        super().__init__(daemon=True)
        self.rq = result_queue
        self.stop_event = threading.Event()
        self.alert = AlertHandler()
        self._prev_time = time.time()

    def run(self):
        while not self.stop_event.is_set():
            item = self.rq.get()
            if item is None: break
            frame, dets = item

            fps = calc_fps(self._prev_time)
            self._prev_time = time.time()

            annotated = draw_detections(frame, dets, fps)

            if dets:
                self.alert.trigger(frame, dets)

            cv2.imshow("Drone Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_event.set()
        self.rq.put(None)