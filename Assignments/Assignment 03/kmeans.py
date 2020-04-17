import numpy as np
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, k, tol=1e-5, max_iter=1000):
        assert ((k > 1) and
                isinstance(k, int)), "k should be a integer and greater than 1"
        self._k = k
        self._centroids = None
        self._tol = tol
        self._max_iter = max_iter

    def fit(self, x):
        if not self._centroids:
            self._centroids = self._init_centroids(x, self._k)
        self._centroids, labels = self._kmeans(x, self._centroids, self._tol,
                                               self._max_iter)
        loss_matrix = np.zeros((len(x), len(self._centroids)))
        for idx, centroid in enumerate(self._centroids):
            loss_matrix[:, idx] = self._compute_dist(x, centroid)
        loss = np.min(loss_matrix, axis=1)
        return self._centroids, labels, loss.sum()

    def predict(self, x):
        dist_matrix = np.zeros((len(x), len(self._centroids)))
        for idx, centroid in enumerate(self._centroids):
            dist_matrix[:, idx] = self._compute_dist(x, centroid)
        labels = np.argmin(dist_matrix, axis=1)
        return labels

    def plot(self, x, labels, centroids):
        for pred in np.unique(labels):
            plt.plot(x[labels == pred][:, 0], x[labels == pred][:, 1], 'o')
        for centroid in centroids:
            plt.plot(centroid[0], centroid[1], '+k')
        plt.title("Predict label with centroids")
        plt.show()

    def _init_centroids(self, x, n):
        centroids = list()
        indexes = list()
        indexes.append(np.random.randint(len(x)))
        centroid_1 = x[indexes[-1]]
        centroids.append(centroid_1)
        dist_array = self._compute_dist(x, centroid_1).tolist()
        indexes.append(np.argmax(dist_array))
        centroids.append(x[indexes[-1]])
        if n > 2:
            for _ in range(2, n):
                dist_array = np.zeros(len(x))
                for centroid in centroids:
                    dist_array += self._compute_dist(x, centroid)
                idx = np.argmax(dist_array)
                while (idx in indexes):
                    dist_array[idx] = 0
                    idx = np.argmax(dist_array)
                indexes.append(idx)
                centroids.append(x[indexes[-1]])
        return centroids

    def _compute_dist(self, x, centroid):
        return np.sum(np.square(x - centroid), axis=1)

    def _l2_norm(self, x):
        x = np.ravel(x, order='K')
        return np.dot(x, x)

    def _kmeans(self, x, centroids, tol, max_iter):
        labels = np.zeros(len(x))
        for _ in range(max_iter):
            dist_matrix = np.zeros((len(x), len(centroids)))
            for idx, centroid in enumerate(centroids):
                dist_matrix[:, idx] = self._compute_dist(x, centroid)
            labels = np.argmin(dist_matrix, axis=1)
            new_centroids = np.zeros_like(centroids)
            for idx in range(len(centroids)):
                new_centroids[idx] = np.mean(x[labels == idx], axis=0)
            shifts = self._l2_norm(new_centroids - centroids)
            if shifts <= tol:
                break
            else:
                centroids = new_centroids

        return centroids, labels
