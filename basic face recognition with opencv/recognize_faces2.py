import cv2

face_recognize = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
captured_frame = None 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('live camera', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        captured_frame = frame.copy()
        
        gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
        faces = face_recognize.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(captured_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('analyzed frame', captured_frame)

    elif key == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()