import numpy as np
import pandas as pd
from sklearn import cluster, metrics


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


df = pd.read_csv("YaleFace.txt", sep=' ', header=None)
df = df.to_numpy()
inputs = df[1:].T
targets = df[0].T
targets = targets.astype(np.int)

K = 15
best_idx = 0
best_loss = float('inf')
best_acc = 0
best_cm = None
for idx in range(10):
    model = cluster.KMeans(K, init="random", n_init=1)
    model.fit(inputs)
    loss, acc, cm = kmeans_metric(model, inputs, targets)
    print(f"{idx+1:02}: Loss={loss:.5f}, Acc={acc:.5f}")
    if (best_loss > loss):
        best_idx = idx + 1
        best_loss = loss
        best_acc = acc
        best_cm = cm
print(
    f"best index: {best_idx:02}, best loss: {best_loss:.5f}, best acc: {best_acc:.5f}"
)
print(f"best confusion matrix:")
print(best_cm)
