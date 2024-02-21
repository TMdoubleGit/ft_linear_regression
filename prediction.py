import numpy as np
import pandas as pd
import joblib


def normalize_data(data, min, max):
    return ((data - min) / (max - min))


def original_data(norm_data, min, max):
    return (norm_data * (max - min) + min)


def coef_determination(y, predictions):
    u = ((y - predictions)**2).sum()
    v = ((y - y.mean())**2).sum()
    return (1 - u / v)


def predict():
    try:
        theta_final = joblib.load("theta_final.pkl")
        predictions = joblib.load("predictions.pkl")
    except FileNotFoundError as e:
        print(e)
        print("Please launch the training program first !")
        exit(1)

    data = pd.read_csv("data.csv")
    x = data['km'].values.reshape(-1, 1)
    y = data['price'].values.reshape(-1, 1)

    norm_y = normalize_data(y, np.min(y), np.max(y))

    mileage = -1
    while (mileage < 0):
        tmp = input("Please, provide your car's mileage: ")
        try:
            mileage = float(tmp)
            if mileage < 0:
                print("Please insert a positive digit value\n")
                continue
        except ValueError:
            print("Please insert a positive digit value\n")
            mileage = -1

    norm_mileage = normalize_data(mileage, np.min(x), np.max(x))
    norm_est_prize = theta_final[1][0] + norm_mileage * theta_final[0][0]
    estimated_price = \
        round(original_data(norm_est_prize, np.min(y), np.max(y)), 2)

    print(f"Best I can do is {estimated_price}$!")
    coef = round(coef_determination(norm_y, predictions), 2)
    print(f"The precision of the algorithm is {coef}")


if __name__ == "__main__":
    predict()
