import cv2

face_recognize = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_recognize.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0),2)
        
    cv2.imshow('live camera', frame)
    
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
    
    