import queue
from pipeline.capture   import CaptureThread
from pipeline.inference import InferenceThread
from pipeline.display   import DisplayThread
import config as cfg

def main():
    frame_q  = queue.Queue(maxsize=cfg.QUEUE_SIZE)
    result_q = queue.Queue(maxsize=cfg.QUEUE_SIZE)

    capture   = CaptureThread(cfg.SOURCE, frame_q)
    inference = InferenceThread(frame_q, result_q)
    display   = DisplayThread(result_q)

    capture.start()
    inference.start()
    display.start()

    display.join()          # block đến khi user bấm 'q'
    capture.stop()
    inference.stop()

if __name__ == "__main__":
    main()