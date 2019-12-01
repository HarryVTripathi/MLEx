
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def plotfun(low, high, error):

    X_range = np.arange(low, high, 0.4)
    Y_range = 20 + 2 * X_range

    X = np.random.randint(low, high, 50)
    Y_obs = 20 + 2 * X + np.random.randint(error * (-1), error, len(X))

    plt.plot(X_range, Y_range)
    plt.scatter(X, Y_obs, color='r', s=12)
    # plt.show()

    X = X.reshape(len(X),1)
    X_test = X_range.reshape((len(X_range),1))

    reg = LinearRegression()
    reg = reg.fit(X, Y_obs)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

    Y_predicted = reg.predict(X_test)

    plt.plot(X_range, Y_range)
    plt.plot(X_test, Y_predicted, 'y')
    plt.scatter(X, Y_obs, color='r', s=12)
    # plt.show()

    return reg.coef_[0], reg.intercept_

b0_arr = np.array([])
b1_arr = np.array([])


for i in range(100):
    b1, b0 = plotfun(200, 800, 100)
    b1_arr = np.insert(b1_arr, 0, b1)
    b0_arr = np.insert(b0_arr, 0, b0)

print(np.mean(b1_arr))
print(np.mean(b0_arr))
