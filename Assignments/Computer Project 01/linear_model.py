import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    '''Linear regression construction function, you can choose vanilla linear 
    regression model or ridge regression model (with l2 regularization)

    @params:
        M (int): order of the polynomial, default is 3
        regularization (None or str): penalty term of the loss function, 
            default is None.
            "None" means vanilla linear regression, there is no penalty term in 
                the loss function
            "l2" means linear regression with l2 regularization or ridge 
                regression, there is a norm-2 term in the loss function
        weight_decay (int or float): only work when regularization is not None, 
            regularization coefficient, default is 1e-3
    '''

    def __init__(self, M=3, regularization=None, weight_decay=1e-3):

        self._dim = M + 1
        if regularization in (None, "None", "none"):
            self._solver = self._solver_vanilla
        elif regularization == "l2":
            self._solver = self._solver_l2
            self._weight_decay = weight_decay
        else:
            raise NotImplementedError(
                f"regularization should be \"None\" or \"l2\", but now is \"{regularization}\""
            )

        self._weights = np.zeros((self._dim, 1))

    @property
    def weight(self):
        '''get the weights of linear regression model

        @return:
            numpy.ndarray: weights of linear regression model
        '''
        return self._weights

    def fit(self, inputs, targets):
        '''fit the regression weights via equations

        @params:
            inputs (numpy.ndarray): training data x or input data
            targets (numpy.ndarray): training data y or target data
        '''
        mat_feature = self._feature_transform(inputs)
        self._weights = self._solver(mat_feature, targets)

    def predict(self, inputs):
        '''use model to generate predictions from input data

        @params:
            inputs (numpy.ndarray): predict data x or input data

        @return:
            numpy.ndarray: predict data for input data
        '''
        mat_x = self._feature_transform(inputs)
        return np.dot(mat_x, self._weights)

    def plot(self,
             viz_range_x=(0, 1),
             viz_range_y=(-1.5, 1.5),
             viz_interval=0.01,
             plot_data=None):
        '''visualize model fitted curve

        @params:
            viz_range_x (array-like): visualize x coordinate ranges, default is 
                (0, 1)
            viz_range_y (array-like): visualize y coordinate ranges, default is 
                (-1.5, 1.5)
            viz_interval (int or float): visualize coordinate interval, default 
                is 0.01
            plot_data (array-like of x and y): plot data point from given data, 
                default is None
        '''
        assert (isinstance(plot_data, list) or isinstance(plot_data, tuple)
               ), f"plot_data should be a list or tuple of inputs and targets"
        viz_data = np.arange(viz_range_x[0], viz_range_x[1], viz_interval)
        plt.plot(viz_data, self.predict(viz_data))
        if plot_data:
            for x, y in zip(plot_data[0], plot_data[1]):
                plt.plot(x, y, 'ro')
        plt.xlim(*viz_range_x)
        plt.ylim(*viz_range_y)



    def _feature_transform(self, inputs):
        '''transform original one-dimensional feature to multi-dimensional 
        features
        
        @params:
            inputs (numpy.ndarray): original one-dimensional data

        @return:
            numpy.ndarray: transformed data matrix
        '''
        mat_x = np.zeros((len(inputs), self._dim))
        for dim in range(self._dim):
            mat_x[:, dim] = np.power(inputs, dim)
        return mat_x

    def _solver_vanilla(self, mat_x, targets):
        '''solve the regression coefficient with vanilla regression and vanilla 
            loss function

        @params:
            mat_x (numpy.ndarray): transformed data matrix
            targets (numpy.ndarray): training data y or target data

        @return:
            numpy.ndarray: solved regression coefficient
        '''
        weights = np.dot(np.dot(np.linalg.inv(np.dot(mat_x.T, mat_x)), mat_x.T),
                         targets)
        return weights

    def _solver_l2(self, mat_x, targets):
        '''solve the regression coefficient with ridge regression and loss 
        function with l2 regularization term

        @params:
            mat_x (numpy.ndarray): transformed data matrix
            targets (numpy.ndarray): training data y or target data

        @return:
            numpy.ndarray: solved regression coefficient
        '''
        alpha = self._weight_decay * np.eye(mat_x.shape[1])
        weights = np.dot(
            np.dot(np.linalg.inv(np.dot(mat_x.T, mat_x) + alpha), mat_x.T),
            targets)
        return weights
