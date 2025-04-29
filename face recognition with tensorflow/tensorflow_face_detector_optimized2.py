import cv2 as cv
from mtcnn import MTCNN

face_recognize = MTCNN()

capture = cv.VideoCapture(0)

captured_frame = None 
processed_frame = None 

while True:
    ret, frame = capture.read()
    if not ret:
        break

    cv.imshow("live camera feed", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord("c"):  
        captured_frame = frame.copy()
        frame_rgb = cv.cvtColor(captured_frame, cv.COLOR_BGR2RGB)
        faces = face_recognize.detect_faces(frame_rgb)

        for face in faces:
            x, y, width, height = face["box"]
            x, y = max(0, x), max(0, y)
            cv.rectangle(captured_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        processed_frame = captured_frame 

    if processed_frame is not None:
        cv.imshow("captured frame", processed_frame)

    if key == ord("q"): 
        break

capture.release()
cv.destroyAllWindows()