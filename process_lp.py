import cv2
import pytesseract
import os
from pathlib import Path

# Đường dẫn đến tệp tesseract.exe trên Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Định nghĩa thư mục đầu vào và thư mục xuất kết quả
input_folder = 'runs/detect/detected/crops/license_plate'  # Thư mục chứa ảnh sau khi khoanh vùng
output_folder = 'processed_images/'  # Thư mục lưu ảnh đã xử lý

# Kiểm tra nếu thư mục kết quả không tồn tại, tạo thư mục mới
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Duyệt qua tất cả các file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    # Kiểm tra nếu file là ảnh (có đuôi .jpg, .png)
    if filename.endswith(('.jpg', '.png')):
        # Đọc ảnh
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # Chuyển ảnh sang xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Làm sạch ảnh
        cleaned_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Binarization
        _, binary_image = cv2.threshold(cleaned_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Phát hiện và cắt ký tự
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lưu các ký tự đã cắt và nhận diện ký tự
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 20:  # Bỏ qua những contour quá nhỏ
                char_image = binary_image[y:y + h, x:x + w]
                char_path = os.path.join(output_folder, f'char_{filename}_{i}.png')
                cv2.imwrite(char_path, char_image)

                # OCR để nhận diện ký tự
                text = pytesseract.image_to_string(char_image,
                                                   config='--psm 10')  # --psm 10 cho OCR nhận diện ký tự đơn lẻ
                print(f"Detected Text for {filename}_{i}: {text}")
