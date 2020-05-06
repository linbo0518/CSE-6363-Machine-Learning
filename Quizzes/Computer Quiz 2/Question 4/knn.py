import numpy as np
import pandas as pd
from utils import distance


class KNN:
    def __init__(self, k_neighbors=5):
        self._k_neighbors = k_neighbors
        self._input_db = None
        self._target_db = None

    def fit(self, inputs, targets):
        assert len(inputs) == len(targets)
        self._input_db = self._check_type(inputs)
        self._target_db = self._check_type(targets)

    def predict(self, inputs):
        inputs = self._check_type(inputs)
        predtions = []
        for x in inputs:
            dist_all = self._calc_distance(x)
            topk_index = np.argsort(dist_all)[:self._k_neighbors]
            topk_targets = self._target_db[topk_index]
            targets, indexes = np.unique(topk_targets, return_inverse=True)
            target_idx = np.bincount(indexes).argmax()
            predtions.append(targets[target_idx])
        return np.array(predtions)

    def _calc_distance(self, x):
        dist_all = []
        for instance in self._input_db:
            dist_all.append(distance(x, instance))
        return dist_all

    def _check_type(self, inputs):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_numpy()
        elif isinstance(inputs, pd.Series):
            inputs = inputs.to_numpy()
        return inputs