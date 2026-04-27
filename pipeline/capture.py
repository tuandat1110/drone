import cv2, threading, time

class CaptureThread(threading.Thread):
    def __init__(self, source, frame_queue):
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.stop_event = threading.Event()
    
    def run(self):
        print(f"Starting video capture from {self.source}")
        cap = cv2.VideoCapture(self.source)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc video hoặc đã hết video")
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except: pass
            self.frame_queue.put(frame)
        cap.release()

    def stop(self):
        self.stop_event.set()