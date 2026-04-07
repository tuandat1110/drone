MODEL_PATH   = "models/best_7.pt"
SOURCE       = "video/drone3.mp4"   # hoặc 0 cho webcam
CONF_THRESHOLD = 0.25

CONF_SMALL   = 0.30   # area < 1500
CONF_LARGE   = 0.50
AREA_THRESH  = 1500
IMGSZ        = 960
DRONE_CLS    = 0

QUEUE_SIZE   = 2      # nhỏ = latency thấp, không tích tụ frame cũ