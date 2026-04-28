# main.py
import asyncio
import queue
import threading
import cv2
from pipeline.capture           import CaptureThread
from pipeline.inference         import InferenceThread
from pipeline.display           import DisplayProcessor
from pipeline.webrtc_publisher  import WebRTCPublisherThread
import config as cfg

def main():
    frame_q     = queue.Queue(maxsize=cfg.QUEUE_SIZE)
    result_q    = queue.Queue(maxsize=cfg.QUEUE_SIZE)
    annotated_q = queue.Queue(maxsize=4)

    capture   = CaptureThread(cfg.SOURCE, frame_q)
    inference = InferenceThread(frame_q, result_q)
    processor = DisplayProcessor()

    # Truyền backend_url + device_key thay vì signaling_url
    webrtc = WebRTCPublisherThread(
        annotated_q=annotated_q,
        backend_url=cfg.BACKEND_URL,   # 'http://192.168.1.100:3001'
        device_key=cfg.DEVICE_KEY,     # 'JETSON_A3F9K2' — đọc từ .env
    )

    capture.start()
    inference.start()
    webrtc.start()

    print("Hệ thống đang chạy. Nhấn 'q' để thoát.")
    print(cfg.BACKEND_URL)
    print(cfg.DEVICE_KEY)

    try:
        while True:
            if not result_q.empty():
                item = result_q.get()
                annotated_frame = processor.process_frame(item)

                if annotated_frame is not None:
                    cv2.imshow("Drone Detection", annotated_frame)
                    try:
                        annotated_q.put_nowait(annotated_frame.copy())
                    except queue.Full:
                        pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        inference.stop()
        webrtc.stop()
        capture.join()
        inference.join()
        webrtc.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()