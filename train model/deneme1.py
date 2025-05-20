import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# === PARAMETRELER ===
model_path = "mobilenetV2_model.h5"  # Model dosyasının yolu
class_names = ["without", "with", "incorrect"]  # Sınıf isimleri

# === MODELİ YÜKLE ===
model = tf.keras.models.load_model(model_path)

# === KAMERAYI AÇ ===
cap = cv2.VideoCapture(0)
print("Kamera açıldı. 'c' ile tahmin yap, 'q' ile çık.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    cv2.imshow("Mask Detector - Press 'c' to capture, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Görüntüyü işleyip tahmin yap
        img = cv2.resize(frame, (224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        print(f"Tahmin: {predicted_class} (Güven: %{confidence * 100:.2f})")

    elif key == ord('q'):
        break

# === TEMİZLEME ===
cap.release()
cv2.destroyAllWindows()
