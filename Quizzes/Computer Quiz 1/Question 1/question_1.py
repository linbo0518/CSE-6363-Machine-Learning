import numpy as np
import matplotlib.pyplot as plt
from utils import rndgenerator, gen_random_data
from linear_model import LinearRegression

MAV_ID = 1001778270
N = 18

# Question 1
x_gen, y_gen = rndgenerator(MAV_ID, N)
x_ori, y_ori = gen_random_data(size=10000, noise=False, func=np.cos)
plt.plot(x_ori, y_ori, ',b')
plt.plot(x_gen, y_gen, 'xr')
plt.show()

# Question 2
for idx, M in enumerate((0, 1, 10, 16, 17), 1):
    model = LinearRegression(M)
    model.fit(x_gen, y_gen)
    plt.subplot(3, 2, idx)
    plt.plot(x_ori, y_ori, ',m')
    model.plot(plot_data=(x_gen, y_gen))
    plt.xlabel(f"M = {M}")
    plt.tight_layout()
    print(f"M = {M}")
    print(f"{model.weight}")
plt.show()

# Question 3
for idx, weight_decay in enumerate((1e-1000, 1e-100, 1e-10, 1e-1, 1), 1):
    model = LinearRegression(17, 'l2', weight_decay=weight_decay)
    model.fit(x_gen, y_gen)
    plt.subplot(3, 2, idx)
    plt.plot(x_ori, y_ori, ',m')
    model.plot(plot_data=(x_gen, y_gen))
    plt.xlabel(f"lambda = {weight_decay}")
    plt.tight_layout()
    print(f"lambda = {weight_decay}")
    print(f"{model.weight}")
plt.show()
