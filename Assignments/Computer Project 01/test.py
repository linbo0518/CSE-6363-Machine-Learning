import matplotlib.pyplot as plt

from utils import gen_random_data
from linear_model import LinearRegression
from naive_bayes import NaiveBayes

# Computer Project 1A
original_x, original_y = gen_random_data(size=10000, noise=False)
inputs, targets = gen_random_data()
plt.plot(original_x, original_y, ',b')
plt.plot(inputs, targets, 'xr')
plt.show()

# Computer Project 1B
for idx, M in enumerate((0, 1, 3, 9), 1):
    model = LinearRegression(M)
    model.fit(inputs, targets)
    plt.subplot(2, 2, idx)
    plt.plot(original_x, original_y, ',m')
    model.plot(plot_data=(inputs, targets))
plt.show()

# Computer Project 1C
for idx, weight_decay in enumerate((1e-8, 1e-6, 1e-4, 1e-2), 1):
    model = LinearRegression(9, 'l2', weight_decay=weight_decay)
    model.fit(inputs, targets)
    plt.subplot(2, 2, idx)
    plt.plot(original_x, original_y, ',m')
    model.plot(plot_data=(inputs, targets))
plt.show()

# Computer Project 1D
model = NaiveBayes()
model.fit('vertebrate.txt', ignore_col="Name")
model.predict('vertebrate.txt', verbose=True)
