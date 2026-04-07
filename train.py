from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('models/yolov8n.pt')
    result = model.train(
        data='Datasets/data.yaml',
        epochs=100,
        patience=25, # Dừng sớm nếu sau 25 epoch không có cải thiện 
        imgsz=640,  # Giữ 640 để nhận diện chi tiết vân sóng trên màn hình
        batch=8,
        device='cpu'    # Thay bằng 'cpu' nếu máy không có GPU Nvidia
    )