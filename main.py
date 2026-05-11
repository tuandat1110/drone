# main.py
import queue
import threading
import time
import traceback
import cv2

from pipeline.capture import CaptureThread
from pipeline.inference import InferenceThread
from pipeline.display import DisplayProcessor
from pipeline.webrtc_publisher import WebRTCPublisherThread
from pipeline.mqtt_publisher import MQTTPublisherThread
import config as cfg


def main():
    # =========================
    # Queue init
    # =========================
    frame_q = queue.Queue(maxsize=cfg.QUEUE_SIZE)
    result_q = queue.Queue(maxsize=cfg.QUEUE_SIZE)
    annotated_q = queue.Queue(maxsize=4)
    stats_q = queue.Queue(maxsize=10)

    # =========================
    # Thread init
    # =========================
    capture = CaptureThread(cfg.SOURCE, frame_q)
    inference = InferenceThread(frame_q, result_q)
    processor = DisplayProcessor()

    webrtc = WebRTCPublisherThread(
        annotated_q=annotated_q,
        backend_url=cfg.BACKEND_URL,
        device_key=cfg.DEVICE_KEY
    )

    mqtt_pub = MQTTPublisherThread(stats_q=stats_q)

    try:
        print("[MAIN] Starting threads...")

        capture.start()
        inference.start()
        webrtc.start()
        mqtt_pub.start()

        print("[MAIN] All threads started.")

        while True:
            try:
                # =========================
                # Get result from inference
                # =========================
                item = result_q.get(timeout=0.03)

            except queue.Empty:
                # Không có frame mới
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[MAIN] Quit by keyboard.")
                    break

                time.sleep(0.005)
                continue

            except Exception as e:
                print("[MAIN] Error getting result_q:", e)
                traceback.print_exc()
                continue

            try:
                # =========================
                # Process frame
                # =========================
                annotated_frame, fps = processor.process_frame(item)

            except Exception as e:
                print("[MAIN] Error processing frame:", e)
                traceback.print_exc()
                continue

            try:
                if annotated_frame is not None:
                    # Show local window
                    cv2.imshow("Drone Detection", annotated_frame)

                    # =========================
                    # Push WebRTC frame
                    # =========================
                    try:
                        annotated_q.put_nowait(annotated_frame.copy())
                    except queue.Full:
                        pass
                    except Exception as e:
                        print("[MAIN] WebRTC queue error:", e)

                    # =========================
                    # Push MQTT stats
                    # =========================
                    try:
                        tmpItem = {
                            "frame": item[0],
                            "detections": item[1],
                            "drone_count": len(item[1]),
                            "bounding_boxes": [d["box"] for d in item[1]],
                            "fps": round(fps, 1),
                        }
                        stats_q.put_nowait({
                            "fps": tmpItem.get("fps", 0),
                            "drone_count": tmpItem.get("drone_count", 0),
                            "bounding_boxes": tmpItem.get("bounding_boxes", [])
                        })
                    except queue.Full:
                        pass
                    except Exception as e:
                        print("[MAIN] MQTT queue error:", e)

            except Exception as e:
                print("[MAIN] Error display frame:", e)
                traceback.print_exc()

            # =========================
            # Quit key
            # =========================
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[MAIN] Quit by keyboard.")
                break

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt received.")

    except Exception as e:
        print("[MAIN] Fatal Error:", e)
        traceback.print_exc()

    finally:
        print("[MAIN] Stopping threads...")

        try:
            capture.stop()
        except:
            pass

        try:
            inference.stop()
        except:
            pass

        try:
            webrtc.stop()
        except:
            pass

        try:
            mqtt_pub.stop()
        except:
            pass

        try:
            capture.join(timeout=2)
        except:
            pass

        try:
            inference.join(timeout=2)
        except:
            pass

        try:
            webrtc.join(timeout=2)
        except:
            pass

        try:
            mqtt_pub.join(timeout=2)
        except:
            pass

        cv2.destroyAllWindows()

        print("[MAIN] Clean exit.")


if __name__ == "__main__":
    main()