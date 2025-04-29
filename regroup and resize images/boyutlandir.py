import os
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np

# Girdi ve çıktı klasörleri
original_path = r"D:\final_dataset"
output_path = r"D:\final_dataset_hazir"

# Tüm sınıfları taramaya yarıyor
for class_folder in os.listdir(original_path):
    class_path = os.path.join(original_path, class_folder)

    if os.path.isdir(class_path):
        output_class_path = os.path.join(output_path, class_folder)
        os.makedirs(output_class_path, exist_ok=True)

        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)

            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = image.load_img(file_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = img_array / 255.0
                    img_array = (img_array * 255).astype(np.uint8)

                    output_file_path = os.path.join(output_class_path, file_name)
                    cv2.imwrite(output_file_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    print(f"{output_file_path} kaydedildi.")
                except Exception as e:
                    print(f"Hata oluştu: {file_name} - {e}")
