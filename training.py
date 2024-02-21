import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def normalize_data(data, min, max):
    return ((data - min) / (max - min))


def original_data(norm_data, min, max):
    return (norm_data * (max - min) + min)


def model(X, theta):
    return X.dot(theta)


def grad(X, y, theta):
    m = len(y)
    return (1/m * X.T.dot(model(X, theta) - y))


def gradient_descent(X, y, theta, learning_rate, n_iteration):
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta


def training():
    data = pd.read_csv("data.csv")
    x = data['km'].values.reshape(-1, 1)
    y = data['price'].values.reshape(-1, 1)

    norm_x = normalize_data(x, np.min(x), np.max(x))
    norm_y = normalize_data(y, np.min(y), np.max(y))

    X = np.hstack((norm_x, np.ones(norm_x.shape)))

    theta = np.zeros((2, 1))

    theta_final = \
        gradient_descent(X, norm_y, theta,
                         learning_rate=0.001, n_iteration=100000)

    predictions = model(X, theta_final)

    plt.scatter(norm_x, norm_y)
    plt.plot(norm_x, predictions, c='r')

    plt.xlabel("Mileage (km)")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1],
               ["0", "50k", "100k", "150k", "200k", "250k"])

    plt.ylabel("Price ($)")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],
               ["0", "1700", "3400", "5100", "6800", "8500"])

    plt.show()

    joblib.dump(theta_final, "theta_final.pkl")
    joblib.dump(predictions, "predictions.pkl")


if __name__ == "__main__":
    training()
