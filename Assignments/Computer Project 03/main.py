import sys
import numpy as np
import pandas as pd
from sklearn import cluster, metrics
np.set_printoptions(threshold=sys.maxsize)


def kmeans_metric(model, inputs, targets, eps=1e-7):
    loss = 0
    clusters = model.cluster_centers_
    predict_mapper = dict()
    preds = model.predict(inputs)
    unique_cluster = np.unique(preds)
    for cluster in unique_cluster:
        loss += np.linalg.norm(inputs[preds == cluster] - clusters[cluster],
                               ord=2)
        predict_mapper[cluster] = np.bincount(
            targets[preds == cluster]).argmax()

    mapped_preds = np.zeros_like(preds)
    for idx, pred in enumerate(preds):
        mapped_preds[idx] = predict_mapper[pred]

    correct = (mapped_preds == targets).sum()
    total = len(targets)
    acc = correct / (total + eps)

    cm = metrics.confusion_matrix(targets, mapped_preds)
    return loss, acc, cm


df = pd.read_csv("ATNTFaceImages.txt", header=None)
df_matrix = df.to_numpy()
inputs = df_matrix[1:].T
targets = df_matrix[0].T

k = 40
model = cluster.KMeans(k)
model.fit(inputs)
loss, acc, cm = kmeans_metric(model, inputs, targets)
print(f"Loss: {loss:.5f} Acc: {acc:.5f}")
print("confusion matrix:")
print(cm)

df = pd.read_csv("HandWrittenLetters.txt", header=None)
df_matrix = df.to_numpy()
inputs = df_matrix[1:].T
targets = df_matrix[0].T

k = 26
model = cluster.KMeans(k)
model.fit(inputs)
loss, acc, cm = kmeans_metric(model, inputs, targets)
print(f"Loss: {loss:.5f} Acc: {acc:.5f}")
print("confusion matrix:")
print(cm)
