import numpy as np
import pandas as pd

from knn import KNN
from centroid_method import CentroidMethod
from svm import SVM
from utils import convert

MAVID = 1001778270

df = pd.read_csv("NBA_0.txt", sep=' ')
player = df.pop("Player").to_numpy()
label = df.pop("Pos").to_numpy()
df = df.to_numpy()
df = convert(df)
label = np.expand_dims(label, 1)
player = np.expand_dims(player, 1)
converted_df = np.concatenate((player, label, df), axis=1)
converted_df = pd.DataFrame(converted_df)
converted_df.to_csv("converted.txt",
                    sep=' ',
                    float_format="%.5f",
                    header=False,
                    index=False)
train_x = df[:482]
train_y = label[:482]
train_y = np.squeeze(train_y)
test = df[482:]

knn_model = KNN(k_neighbors=3)
centroid_model = CentroidMethod()
svm_model = SVM(C=1.0, kernel="linear")
knn_model.fit(train_x, train_y)
knn_pred = knn_model.predict(test)
print(f"KNN Result: {knn_pred}")
centroid_model.fit(train_x, train_y)
centroid_pred = centroid_model.predict(test)
print(f"Centroid Method result: {centroid_pred}")
svm_model.fit(train_x, train_y)
svm_pred = svm_model.predict(test)
print(f"SVM Result: {svm_pred}")