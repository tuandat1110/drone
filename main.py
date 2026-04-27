import asyncio
import queue
import threading
import cv2
from pipeline.capture   import CaptureThread
from pipeline.inference import InferenceThread
from pipeline.display   import DisplayProcessor
from pipeline.webrtc_publisher import WebRTCPublisherThread  # Thêm mới
import config as cfg

def main():
    frame_q      = queue.Queue(maxsize=cfg.QUEUE_SIZE)
    result_q     = queue.Queue(maxsize=cfg.QUEUE_SIZE)
    annotated_q  = queue.Queue(maxsize=4)  # Buffer nhỏ thôi, WebRTC cần low-latency

    capture   = CaptureThread(cfg.SOURCE, frame_q)
    inference = InferenceThread(frame_q, result_q)
    processor = DisplayProcessor()

    # WebRTC thread mới — tự chạy asyncio loop bên trong
    webrtc = WebRTCPublisherThread(
        annotated_q=annotated_q,
        signaling_url=cfg.SIGNALING_URL,  # 'http://localhost:3000'
    )

    capture.start()
    inference.start()
    webrtc.start()  # Thêm dòng này

    print("Hệ thống đang chạy. Nhấn 'q' để thoát.")

    try:
        while True:
            if not result_q.empty():
                item = result_q.get()
                annotated_frame = processor.process_frame(item)

                if annotated_frame is not None:
                    # Hiển thị local như cũ
                    cv2.imshow("Drone Detection", annotated_frame)

                    # Đẩy vào queue cho WebRTC (non-blocking, drop nếu full)
                    try:
                        annotated_q.put_nowait(annotated_frame.copy())
                    except queue.Full:
                        pass  # Bỏ qua frame nếu WebRTC chưa kịp xử lý

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        inference.stop()
        webrtc.stop()      # Thêm dòng này
        capture.join()
        inference.join()
        webrtc.join()      # Thêm dòng này
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()