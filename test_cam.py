import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"Camera OK! Resolution: {frame.shape}")
        cv2.imwrite("test_frame.jpg", frame)  # Lưu ảnh để kiểm tra
    else:
        print("Mở được camera nhưng không đọc được frame")
    cap.release()
else:
    print("Không tìm thấy camera")