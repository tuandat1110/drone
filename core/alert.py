import cv2
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertHandler:
    """
    Xử lý cảnh báo khi phát hiện drone.

    Tính năng:
      - Log cảnh báo ra console / file
      - Lưu snapshot frame lúc phát hiện
      - Cooldown để tránh spam alert liên tục
    """

    def __init__(self,
                 snapshot_dir: str = "output/snapshots",
                 cooldown_sec: float = 3.0,
                 save_snapshot: bool = True):
        self.snapshot_dir   = Path(snapshot_dir)
        self.cooldown_sec   = cooldown_sec
        self.save_snapshot  = save_snapshot
        self._last_alert_at = 0.0

        if self.save_snapshot:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [ALERT] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def trigger(self, frame, detections: list):
        """
        Gọi mỗi khi có detection. Chỉ thực sự alert sau cooldown.
        """
        now = time.time()
        if now - self._last_alert_at < self.cooldown_sec:
            return

        self._last_alert_at = now
        count = len(detections)

        logger.info(f"{count} drone(s) detected.")

        if self.save_snapshot:
            self._save_snapshot(frame)

        # --- Mở rộng tại đây ---
        # self._send_webhook(detections)
        # self._send_email(detections)
        # self._push_notification(detections)

    def _save_snapshot(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.snapshot_dir / f"drone_{timestamp}.jpg"
        cv2.imwrite(str(path), frame)
        logger.info(f"Snapshot saved → {path}")