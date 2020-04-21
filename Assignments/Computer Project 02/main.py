import numpy as np
import pandas as pd
from knn import KNN
from centroid_method import CentroidMethod
from svm import SVM
from utils import convert
df = pd.read_csv('american_people_1000.txt', sep=' ')
df = df.drop(labels=["fnlwgt", "EducationNum", "CapitalGain", "CapitalLoss"],
             axis=1)
label = df.pop("Income")
x_train, y_train = df[:900], label[:900]
x_test, y_test = df[900:], label[900:]

# K Nearest Neighbor
model = KNN()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("K Nearest Neighbor")
print(f"Acc: {(pred == y_test).sum() / len(pred)}")

df = convert(df.to_numpy())
label = label.to_numpy()
x_train, y_train = df[:900], label[:900]
x_test, y_test = df[900:], label[900:]

# Centroid Method
model = CentroidMethod()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("Centroid Method")
print(f"Acc: {(pred == y_test).sum() / len(pred)}")

model = SVM(kernel='linear')
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("SVM with Linear Kernel")
print(f"Acc: {(pred == y_test).sum() / len(pred)}")

# SVM with Gaussian Kernel
model = SVM(kernel='rbf')
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("SVM with Gaussian Kernel")
print(f"Acc: {(pred == y_test).sum() / len(pred)}")
