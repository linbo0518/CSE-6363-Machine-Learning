import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans

SIGMA_1 = (3, 3)
SIGMA_2 = (-3, 3)
SIGMA_3 = (0, -3)
MU = ((4, 0), (0, 4))
SIZE = 100

dist_1 = np.random.multivariate_normal(SIGMA_1, MU, size=SIZE)
dist_2 = np.random.multivariate_normal(SIGMA_2, MU, size=SIZE)
dist_3 = np.random.multivariate_normal(SIGMA_3, MU, size=SIZE)

plt.plot(dist_1[:, 0], dist_1[:, 1], 'o')
plt.plot(dist_2[:, 0], dist_2[:, 1], 'o')
plt.plot(dist_3[:, 0], dist_3[:, 1], 'o')
plt.title("Training data (sigma = 2)")
plt.show()

x = np.concatenate((dist_1, dist_2, dist_3), axis=0)

for i in range(5):
    model = KMeans(3)
    centroids, labels, loss = model.fit(x)
    model.plot(x, labels, centroids)
    print("Centroids:")
    for centroid in centroids:
        print(f"\t{centroid}")
    print(f"KMeans Loss: {loss:.7f}")

MU = ((16, 0), (0, 16))
dist_1 = np.random.multivariate_normal(SIGMA_1, MU, size=SIZE)
dist_2 = np.random.multivariate_normal(SIGMA_2, MU, size=SIZE)
dist_3 = np.random.multivariate_normal(SIGMA_3, MU, size=SIZE)

plt.plot(dist_1[:, 0], dist_1[:, 1], 'o')
plt.plot(dist_2[:, 0], dist_2[:, 1], 'o')
plt.plot(dist_3[:, 0], dist_3[:, 1], 'o')
plt.title("Training data (sigma = 4)")
plt.show()

x = np.concatenate((dist_1, dist_2, dist_3), axis=0)

for i in range(5):
    model = KMeans(3)
    centroids, labels, loss = model.fit(x)
    model.plot(x, labels, centroids)
    print("Centroids:")
    for centroid in centroids:
        print(f"\t{centroid}")
    print(f"KMeans Loss: {loss:.7f}")