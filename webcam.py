import cv2

from imageai.Prediction.Custom import CustomImagePrediction
from time import sleep
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
    return frame


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:

        ret_val, img = cam.read()

        if mirror:
            img = cv2.flip(img, 1)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

        cv2.imwrite("templates/webcam-image.jpg", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = detect(gray, img)

        predictions, probabilities = prediction.predictImage("templates/webcam-image.jpg", result_count=2)
        if probabilities[0] < 80:
            cv2.rectangle(img, (0, 0), (50, 50), (0, 255, 0), -1)
        else:
            cv2.rectangle(img, (0, 0), (50, 50), (0, 0, 255), -1)

        #cv2.imshow('Webcam Feed', img)

        print("Frown: ", probabilities[0], " Smile: ", probabilities[1])

        cv2.imwrite("templates/ml-image.jpg", img)
        sleep(1/30)

    cv2.destroyAllWindows()


execution_path = os.getcwd()
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("datasets/models/model_ex-062_acc-0.916385.h5")
prediction.setJsonPath("datasets/json/model_class.json")
prediction.loadModel(num_objects=2)

show_webcam(mirror=True)
