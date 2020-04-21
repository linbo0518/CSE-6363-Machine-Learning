import numpy as np
from utils import distance


class CentroidMethod:
    def __init__(self):
        self._centroids = None
        self._unique_target = None

    def fit(self, inputs, targets):
        self._unique_target = np.unique(targets)
        self._centroids = np.zeros((len(self._unique_target), inputs.shape[1]))
        for idx, target in enumerate(self._unique_target):
            self._centroids[idx] = np.mean(inputs[targets == target], axis=0)

    def predict(self, inputs):
        predtions = []
        for x in inputs:
            dist_all = self._calc_distance(x)
            predtions.append(self._unique_target[np.argmin(dist_all)])
        return np.array(predtions)

    def _calc_distance(self, x):
        dist_all = []
        for centroid in self._centroids:
            dist_all.append(distance(x, centroid))
        return dist_all
