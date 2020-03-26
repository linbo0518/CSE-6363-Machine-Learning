import numpy as np
import math


def rndgenerator(seed, n):
    pi = math.pi
    x = np.linspace(0, 1, n)
    y = np.cos(2 * pi * x)

    np.random.seed(seed)
    e = np.random.random([n])

    for i in range(0, n):
        y[i] = y[i] + ((-1)**i) * (0.1 + e[i])

    return x, y


def gen_random_data(size=10,
                    data_range=(0, 1),
                    noise=True,
                    noise_range=(-0.05, 0.05),
                    x_coef=2 * np.pi,
                    func=np.sin):
    '''Generate random data from a given function.
    
    @params:
        size (int): number of data generated, default is 10.
        data_range (array-like): data range from low to high, defaulr is (0, 1)
        noise (bool): whether to add noise to the generated data, default is 
            True.
        nosie_range (array-like): noise range from low to high, invalid if s is 
            false, default is (-0.05, 0.05)
        coefficient (int or float): coefficient of input data, defaule is 
            2 * np.pi
        func (function): callback function that can give output data from input 
            data. default is np.sin

    @return:
        numpy.ndarray: generated input data X
        numpy.ndarray: generated output data Y
    '''
    assert (data_range[0] <= data_range[1]), \
        f"\"data_range\" should be a array-like from low to high, but now it is {data_range}"
    assert (noise_range[0] <= noise_range[1]), \
        f"\"noise_range\" should be a array-like from low to high, but now it is {noise_range}"

    data_coef = data_range[1] - data_range[0]
    data_bias = (data_range[0] + data_range[1]) / 2 - data_coef / 2
    xs = np.random.random(size) * data_coef + data_bias
    ys = np.zeros(size)

    if noise:
        noise_coef = noise_range[1] - noise_range[0]
        noise_bias = (noise_range[0] + noise_range[1]) / 2 - noise_coef / 2

    for idx, x in enumerate(xs):
        ys[idx] = func(x_coef * x)
        if noise:
            xs[idx] += np.random.random() * noise_coef + noise_bias
    return xs, ys