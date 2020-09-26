import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect(frame):
    frame = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame, frame[y:y + h, x:x + w]
