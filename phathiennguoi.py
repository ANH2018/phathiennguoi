import cv2
import csv
import winsound
import os

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r'\haarcascade_frontalface_default.xml')


# Mở camera
cap = cv2.VideoCapture(0)  # Số 0 thường là camera mặc định

# Kiểm tra xem camera có được mở không
if not cap.isOpened():
    print("Không thể mở camera. Hãy kiểm tra lại.")
    exit()

# Thư mục để lưu hình ảnh
save_dir = r'F:\Nucleo\imga'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Tệp tin CSV để lưu vị trí của khuôn mặt
csv_file = open('face_positions.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'X', 'Y', 'Width', 'Height'])  # Header của tệp tin CSV

# Biến kiểm tra xem khuôn mặt đã được phát hiện hay chưa
face_detected = False

# Biến để đếm số lượng hình ảnh đã được lưu
image_count = 0

# Biến để đếm số lượng frame đã xử lý
frame_count = 0

# Vòng lặp để hiển thị dữ liệu video từ camera
while True:
    # Đọc frame từ camera
    ret, frame = cap.read()

    # Kiểm tra xem frame có được đọc thành công không
    if not ret:
        print("Không thể nhận frame từ camera.")
        break

    # Chuyển đổi frame sang ảnh xám để tăng tốc độ xử lý
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Nếu phát hiện khuôn mặt
    if len(faces) > 0:
        # Nếu khuôn mặt đã được phát hiện trước đó, không phát âm thanh
        if not face_detected:
            winsound.Beep(1000, 1000)  # Phát âm thanh kiểu chu kỳ, tần số 1000Hz, thời gian 1 giây
            face_detected = True

        # Lưu vị trí của khuôn mặt vào tệp tin CSV
        for (x, y, w, h) in faces:
            csv_writer.writerow([frame_count, x, y, w, h])

        # Lưu hình ảnh chứa khuôn mặt
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            image_path = os.path.join(save_dir, f"face_{image_count}.jpg")
            cv2.imwrite(image_path, face_image)
            image_count += 1
    else:
        face_detected = False

    # Vẽ hình chữ nhật xung quanh khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Hiển thị frame
    cv2.imshow('Face Detection', frame)

    # Tăng biến đếm số lượng frame đã xử lý
    frame_count += 1

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng tệp tin CSV
csv_file.close()

# Giải phóng camera và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
