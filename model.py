from imageai.Prediction.Custom import CustomImagePrediction
from detect import detect


class Model(object):
    def __init__(self):
        self.prediction = CustomImagePrediction()
        self.prediction.setModelTypeAsResNet()
        self.prediction.setModelPath("datasets/models/model_ex-062_acc-0.916385.h5")
        self.prediction.setJsonPath("datasets/json/model_class.json")
        self.prediction.loadModel(num_objects=2)
        self.webcam = Webcam()

    def predict(self, frame):
        pred, prob = self.prediction.predictImage(detect(frame)[1], input_type="array", result_count=2)

        if prob[0] < 80:
            cv2.rectangle(frame, (0, 0), (50, 50), (0, 255, 0), -1)
        else:
            cv2.rectangle(frame, (0, 0), (50, 50), (0, 0, 255), -1)

        return frame
