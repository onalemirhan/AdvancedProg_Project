import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("mobilenetv2_model.h5")  
detector = MTCNN()

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb_frame[y:y+height, x:x+width]

        if face_crop.size == 0:
            continue

        face_resized = cv2.resize(face_crop, (224, 224))
        face_array = preprocess_input(face_resized.astype(np.float32))
        face_array = np.expand_dims(face_array, axis=0)

        prediction = model.predict(face_array)[0]
        class_index = np.argmax(prediction)
        class_names = ["Incorrect", "With Mask", "Without Mask"]
        label = class_names[class_index]
        confidence = prediction[class_index] * 100 

        if label == "Incorrect":
            box_color = (255, 0, 0)  
        elif label == "With Mask":
            box_color = (0, 255, 0)  
        else:  
            box_color = (0, 0, 255)  

        text = f"{label} {confidence:.1f}%"
        cv2.rectangle(frame, (x, y), (x+width, y+height), box_color, 2)
        cv2.putText(frame, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, box_color, 2)

    return frame

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)
        result = process_frame(img)
        cv2.imshow("Prediction", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def open_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)

        if key == ord("c"):
            result_frame = process_frame(frame.copy())
            if result_frame is not None:
                cv2.imshow("Prediction", result_frame)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mask Face Detector")
        self.geometry("400x300")

        self.label = ctk.CTkLabel(self, text="Mask Face Detector", font=("Arial", 20))
        self.label.pack(pady=20)

        self.select_button = ctk.CTkButton(self, text="Select Image", command=select_image)
        self.select_button.pack(pady=10)

        self.camera_button = ctk.CTkButton(self, text="Open Camera", command=open_camera)
        self.camera_button.pack(pady=10)

        self.reports_button = ctk.CTkButton(self, text="Reports", command=self.open_reports_window)
        self.reports_button.pack(pady=20)

    def open_reports_window(self):
        self.withdraw()
        ReportsWindow(self)

class ReportsWindow(ctk.CTkToplevel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.title("Reports")
        self.geometry("400x300")

        self.back_button = ctk.CTkButton(self, text="← Back", width=60, command=self.go_back)
        self.back_button.place(x=10, y=10)

        self.export_all_button = ctk.CTkButton(self, text="Export All Reports as PDF", command=self.export_all)
        self.export_all_button.pack(pady=(60, 10))

        self.export_accuracy_button = ctk.CTkButton(self, text="Export Accuracy Summary as PDF", command=self.export_accuracy)
        self.export_accuracy_button.pack(pady=10)

    def go_back(self):
        self.destroy()
        self.main_window.deiconify()

    def export_all(self):
        print("Export All clicked")

    def export_accuracy(self):
        print("Export Accuracy clicked")

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
