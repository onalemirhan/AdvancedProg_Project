import customtkinter as ctk

# Temayı ve font boyutlarını ayarla
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mask Face Detector")
        self.geometry("400x300")

        self.label = ctk.CTkLabel(self, text="Mask Face Detector", font=("Arial", 20))
        self.label.pack(pady=20)

        self.select_button = ctk.CTkButton(self, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.camera_button = ctk.CTkButton(self, text="Open Camera", command=self.open_camera)
        self.camera_button.pack(pady=10)

        self.reports_button = ctk.CTkButton(self, text="Reports", command=self.open_reports_window)
        self.reports_button.pack(pady=20)

    def select_image(self):
        print("Select Image clicked")

    def open_camera(self):
        print("Open Camera clicked")

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
