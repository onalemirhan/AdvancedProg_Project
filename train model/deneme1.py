import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Modeli yükle
model = load_model("mobilenetv2_model.h5")
class_names = ['incorrect_mask', 'with_mask', 'without_mask'] 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press 'c' to capture / 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # tahmin burada oluyor
        prediction = model.predict(img)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[predicted_index] * 100

        print(f"Tahmin edilen sınıf: {predicted_class} ({confidence:.2f}%)")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
