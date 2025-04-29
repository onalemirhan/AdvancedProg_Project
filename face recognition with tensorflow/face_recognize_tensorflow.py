import cv2 as cv
from mtcnn import MTCNN

face_recognize = MTCNN()

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    faces = face_recognize.detect_faces(frame_rgb)

    for face in faces:
        x, y, width, height = face["box"]

        x, y = max(0,x) , max(0,y)
        cv.rectangle(frame, (x,y), (x+width, y+height),(0,255,0),2)
    cv.imshow("tensorflow face detector", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()