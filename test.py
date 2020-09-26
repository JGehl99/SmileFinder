from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("datasets/models/model_ex-062_acc-0.916385.h5")
prediction.setJsonPath("datasets/json/model_class.json")
prediction.loadModel(num_objects=2)

predictions, probabilities = prediction.predictImage("frown.jpg", result_count=2)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
