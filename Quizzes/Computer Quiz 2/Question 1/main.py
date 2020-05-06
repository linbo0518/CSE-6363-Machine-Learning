from svm import SVM
from utils import dataGenerator

MAVID = 1001778270

x, y = dataGenerator(MAVID, 100)
model = SVM(C=1.0, kernel="linear")
model.fit(x.T, y.T)
print(f"1. weight = {model.weight}, bias = {model.bias}")
print(f"2. predict value = {model.weight @ x + model.bias}")
print(f"3. indexes of support vectors: {model._model.support_}")
print(f"4. alpha = {model.alpha}")