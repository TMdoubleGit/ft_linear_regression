import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

x , y = make_regression(n_samples= 100, n_features=1, noise=10)
y = y.reshape(y.shape[0], 1)

X = np.hstack((x, np.ones(x.shape)))

theta = np.random.randn(2, 1)

def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return (1/(2 * m) *np.sum((model(X, theta) - y)**2))

def grad(X, y, theta):
    m = len(y)
    return (1/m * X.T.dot(model(X, theta) - y))

def gradient_descent(X, y, theta, learning_rate, n_iteration):
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

theta_final = gradient_descent(X, y, theta, learning_rate=0.01, n_iteration=10000)

predictions = model(X, theta_final)
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.show()