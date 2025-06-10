import customtkinter as ctk
from tkinter import filedialog
from fpdf import FPDF
import webbrowser
import os
import sqlite3
import cv2
import numpy as np
from mtcnn import MTCNN
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime, timedelta
from tkinter import messagebox

model = load_model("mobilenetv2_model.h5")  
detector = MTCNN()


conn = sqlite3.connect(r"C:\Users\emirh\OneDrive\Masaüstü\proje main\advancedprojectdb.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS Data (
        date TEXT,
        prediction TEXT,
        confidence NUMERIC
    )
""")

conn.commit()
conn.close()

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

        # ✅ Tahmini veritabanına kaydet
        save_prediction_to_db(label, float(confidence))

        # ✅ Kutuyu çiz ve yazıyı yaz
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
    
def save_prediction_to_db(prediction, confidence):
    conn = sqlite3.connect("advancedprojectdb.db")
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS Data ( date TEXT, prediction TEXT, confidence NUMERIC)""")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO Data (date, prediction, confidence) VALUES (?, ?, ?)",(now, prediction, confidence))
    conn.commit()
    conn.close

def export_all():
    conn = sqlite3.connect("advancedprojectdb.db")
    cursor = conn.cursor()

    cursor.execute("SELECT date, prediction, confidence FROM Data")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No data to export.")
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="All Test Results", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    for row in rows:
        date, prediction, confidence = row
        pdf.cell(0, 10, txt=f"Date: {date} | Prediction: {prediction} | Confidence: {confidence:.2f}", ln=True)
    pdf_path = "all_test_results.pdf"
    pdf.output(pdf_path)
    abs_path = os.path.abspath(pdf_path)
    webbrowser.open(f"file://{abs_path}")

def export_accuracy():
    conn = sqlite3.connect("advancedprojectdb.db")
    cursor = conn.cursor()

    cursor.execute("SELECT date, prediction FROM Data")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No data to export.")
        return

    all_preds = [row[1] for row in rows]
    all_counts = Counter(all_preds)

    now = datetime.now()
    one_month_ago = now - timedelta(days=30)
    one_week_ago = now - timedelta(days=7)

    def filter_by_date(limit_date):
        return [row[1] for row in rows if datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") >= limit_date]

    last_month_preds = filter_by_date(one_month_ago)
    last_week_preds = filter_by_date(one_week_ago)

    def get_percentages(pred_list):
        count = Counter(pred_list)
        total = sum(count.values())
        if total == 0:
            return {"With Mask": 0, "Without Mask": 0, "Incorrect": 0}
        return {
            "With Mask": round((count["With Mask"] / total) * 100, 2),
            "Without Mask": round((count["Without Mask"] / total) * 100, 2),
            "Incorrect": round((count["Incorrect"] / total) * 100, 2)
        }

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Mask Usage Accuracy Report", ln=True, align="C")
    pdf.ln(10)

    def write_section(title, percentages):
        pdf.set_font("Arial", style='B', size=11)
        pdf.cell(0, 10, txt=title, ln=True)
        pdf.set_font("Arial", size=10)
        for key, val in percentages.items():
            pdf.cell(0, 10, txt=f"{key}: {val}%", ln=True)
        pdf.ln(5)

    write_section("All Time Statistics", get_percentages(all_preds))
    write_section("Last Month Statistics", get_percentages(last_month_preds))
    write_section("Last Week Statistics", get_percentages(last_week_preds))
    month_percent = get_percentages(last_month_preds)["With Mask"]
    week_percent = get_percentages(last_week_preds)["With Mask"]
    change = round(week_percent - month_percent, 2)

    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=11)
    pdf.cell(0, 10, txt="Change in Mask Usage Accuracy", ln=True)
    pdf.set_font("Arial", size=10)
    if change > 0:
        pdf.cell(0, 10, txt=f"Improved by {change}% in the last week compared to last month.", ln=True)
    elif change < 0:
        pdf.cell(0, 10, txt=f"Dropped by {-change}% in the last week compared to last month.", ln=True)
    else:
        pdf.cell(0, 10, txt="No change in mask usage accuracy between last week and last month.", ln=True)

    pdf.set_font("Arial", style='B', size=11)
    pdf.cell(0, 10, txt="Prediction Counts", ln=True)
    pdf.set_font("Arial", size=10)
    for cls in ["With Mask", "Without Mask", "Incorrect"]:
        pdf.cell(0, 10, txt=f"{cls}: {all_counts.get(cls, 0)} times", ln=True)

    pdf_path = "accuracy_report.pdf"
    pdf.output(pdf_path)

    abs_path = os.path.abspath(pdf_path)
    webbrowser.open(f"file://{abs_path}")


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

        self.reports_button = ctk.CTkButton(self, text="Reports", command=self.open_password_prompt)
        self.reports_button.pack(pady=20)

    def open_reports_window(self):
        self.withdraw()
        ReportsWindow(self)    

    def open_password_prompt(self):
        CORRECT_PASSWORD = "1234"
        def verify_password():
            entered = password_entry.get()
            if entered == CORRECT_PASSWORD:
                password_window.destroy()
                self.open_reports_window() 
            else:
                messagebox.showerror("Error", "Incorrect password!")

        password_window = ctk.CTkToplevel()
        password_window.title("Enter Password")
        password_window.geometry("300x150")

        label = ctk.CTkLabel(password_window, text="Enter password to access reports:")
        label.pack(pady=10)

        password_entry = ctk.CTkEntry(password_window, show="*")
        password_entry.pack(pady=5)

        confirm_button = ctk.CTkButton(password_window, text="Confirm", command=verify_password)
        confirm_button.pack(pady=10)

class ReportsWindow(ctk.CTkToplevel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.title("Reports")
        self.geometry("400x300")

        self.back_button = ctk.CTkButton(self, text="← Back", width=60, command=self.go_back)
        self.back_button.place(x=10, y=10)

        self.export_all_button = ctk.CTkButton(self, text="Export All Reports as PDF", command=export_all)
        self.export_all_button.pack(pady=(60, 10))

        self.export_accuracy_button = ctk.CTkButton(self, text="Export Accuracy Summary as PDF", command=export_accuracy)
        self.export_accuracy_button.pack(pady=10)

    def go_back(self):
        self.destroy()
        self.main_window.deiconify()


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
