import cv2
from imageai.Prediction.Custom import CustomImagePrediction
from time import sleep
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def get_frame(gray, img):
    faces = face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(30, 30))

    try:
        f = faces[0]
        return img[f[1]:f[1] + f[3], f[0]:f[0] + f[2]]
    except:
        return img


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)

    frameskip = 5

    while True:

        ret_val, img = cam.read()

        if mirror:
            img = cv2.flip(img, 1)

        if cv2.waitKey(1) == 27:
            break

        if frameskip % 5 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = (gray, img)
            # cv2.imwrite("faceimage.jpg", face)
            predictions, probabilities = prediction.predictImage(face, input_type="array", result_count=2)
            print("Frown: ", probabilities[0], " Smile: ", probabilities[1])

        if probabilities[0] < 80:
            cv2.rectangle(img, (0, 0), (50, 50), (0, 255, 0), -1)
        else:
            cv2.rectangle(img, (0, 0), (50, 50), (0, 0, 255), -1)

        frameskip = (frameskip + 1) % 10
        cv2.imshow('SmileFinder', img)

    cv2.destroyAllWindows()


execution_path = os.getcwd()
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("datasets/models/model_ex-062_acc-0.916385.h5")
prediction.setJsonPath("datasets/json/model_class.json")
prediction.loadModel(num_objects=2)

show_webcam(mirror=True)
