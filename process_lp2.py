import cv2
import pytesseract
import os
from pathlib import Path
import numpy as np

# Đường dẫn đến tệp tesseract.exe trên Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Định nghĩa thư mục đầu vào và thư mục xuất kết quả
input_folder = 'runs/detect/detected/crops/license_plate'
output_folder = 'processed_images/'

# Kiểm tra nếu thư mục kết quả không tồn tại, tạo thư mục mới
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Hàm xoay ảnh dựa trên góc nghiêng
def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Duyệt qua tất cả các file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # Xoay chỉnh ảnh để làm thẳng
        corrected_image = correct_skew(image)

        # Chuyển sang ảnh xám
        gray_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)

        # Làm sạch ảnh
        cleaned_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Binarization
        _, binary_image = cv2.threshold(cleaned_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR để nhận diện toàn bộ biển số
        text = pytesseract.image_to_string(binary_image, config='--psm 6')  # --psm 6 để nhận diện đoạn văn bản
        print(f"Detected Text for {filename}: {text}")

        # Lưu ảnh đã chỉnh sửa
        output_path = os.path.join(output_folder, f'processed_{filename}')
        cv2.imwrite(output_path, binary_image)
