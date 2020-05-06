import numpy as np
import pandas as pd
from sklearn import svm


class SVM:
    def __init__(self, C=1.0, kernel='rbf'):
        self._model = svm.SVC(C=C, kernel=kernel)

    def fit(self, inputs, targets):
        self._model.fit(inputs, targets)

    def predict(self, inputs):
        return self._model.predict(inputs)

    @property
    def weight(self):
        return self._model.coef_

    @property
    def beta(self):
        return self._model.intercept_

    @property
    def alpha(self):
        return abs(self._model.dual_coef_)

    @property
    def margin(self):
        return 2 / np.linalg.norm(self.weight)

    def _check_type(self, inputs):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_numpy()
        elif isinstance(inputs, pd.Series):
            inputs = inputs.to_numpy()
        return inputs