import cv2 as cv
from mtcnn import MTCNN

face_recognize = MTCNN()

capture = cv.VideoCapture(0)

frame_counter = 0

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_counter += 1

    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if frame_counter % 3 == 0:
        frame_rgb = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        faces = face_recognize.detect_faces(frame_rgb)
    else:
        faces = []

    for face in faces:
        x, y, width, height = face["box"]

        x, y, width, height = int(x * 2), int(y * 2), int(width * 2), int(height * 2)

        x, y = max(0, x), max(0, y)

        cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv.imshow("tensorFlow face detector optimized", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()
