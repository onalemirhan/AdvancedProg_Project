import sqlite3
from datetime import datetime

conn = sqlite3.connect("advancedprojectdb.db")
cursor = conn.cursor()

current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

prediction = input("Tahmin sonucunu gir (prediction): ")
confidence = input("Confidence değerini gir (% olarak): ")

cursor.execute("""
    INSERT INTO Data (date, prediction, confidence) 
    VALUES (?, ?, ?)
""", (current_date, prediction, float(confidence)))

conn.commit()
conn.close()

print("Veri başarıyla kaydedildi.")
