# pipeline/mqtt_publisher.py
import paho.mqtt.client as mqtt
import json
import threading
import queue
import time
import os
import ssl

class MQTTPublisherThread(threading.Thread):
    def __init__(self, stats_q: queue.Queue):
        super().__init__(daemon=True)
        self.stats_q     = stats_q
        self.device_key  = os.getenv('DEVICE_KEY', 'JETSON_001')
        self.broker_host = os.getenv('MQTT_BROKER_HOST')
        self.broker_port = int(os.getenv('MQTT_PORT', 8883))
        self.username    = os.getenv('MQTT_USERNAME')
        self.password    = os.getenv('MQTT_PASSWORD')
        self._stop_event = threading.Event()
        self._connected  = False

        self.client = mqtt.Client(
            client_id=self.device_key,
            protocol=mqtt.MQTTv5,
        )
        self.client.username_pw_set(self.username, self.password)

        # TLS — HiveMQ Cloud bắt buộc
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLS)

        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish    = self._on_publish

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._connected = True
            print(f'[MQTT] Connected to HiveMQ Cloud')
            # Publish trạng thái online ngay khi kết nối
            self._publish_heartbeat()
        else:
            print(f'[MQTT] Connect failed, rc={rc}')

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self._connected = False
        print(f'[MQTT] Disconnected, rc={rc}')

    def _on_publish(self, client, userdata, mid):
        pass  # bật nếu cần debug

    def _publish_heartbeat(self):
        self.client.publish(
            topic=f'drone/{self.device_key}/heartbeat',
            payload=json.dumps({
                'device_key': self.device_key,
                'status':     'online',
                'timestamp':  time.time(),
            }),
            qos=1,
        )

    def publish_stats(self, fps: float, drone_count: int, bounding_boxes: list = []):
        if not self._connected:
            return
        self.client.publish(
            topic=f'drone/{self.device_key}/stats',
            payload=json.dumps({
                'fps':            round(fps, 1),
                'drone_count':    drone_count,
                'bounding_boxes': bounding_boxes,
                'timestamp':      time.time(),
            }),
            qos=1,
        )

    def run(self):
        self.client.connect(self.broker_host, port=self.broker_port, keepalive=60)
        self.client.loop_start()

        last_heartbeat = time.time()

        while not self._stop_event.is_set():
            try:
                stats = self.stats_q.get(timeout=1.0)
                self.publish_stats(
                    fps=stats['fps'],
                    drone_count=stats['drone_count'],
                    bounding_boxes=stats.get('bounding_boxes', []),
                )
            except queue.Empty:
                pass

            # Heartbeat mỗi 30 giây
            if time.time() - last_heartbeat > 30:
                self._publish_heartbeat()
                last_heartbeat = time.time()

    def stop(self):
        # Publish offline trước khi ngắt
        if self._connected:
            self.client.publish(
                topic=f'drone/{self.device_key}/heartbeat',
                payload=json.dumps({
                    'device_key': self.device_key,
                    'status':     'offline',
                    'timestamp':  time.time(),
                }),
                qos=1,
                retain=True,
            )
            time.sleep(0.5)  # chờ gửi xong
        self._stop_event.set()
        self.client.loop_stop()
        self.client.disconnect()