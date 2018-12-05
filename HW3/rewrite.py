import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns

def load_file(filename):
    df = pd.read_csv(filename)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # split train and test
    dataNumber = int(len(df.index) * 0.8)
    train_data = df[:dataNumber]
    test_data = df[dataNumber + 1:].reset_index(drop=True)

    return train_data, test_data

def MSE(y, Y):
    return np.sum(np.square(y - Y)) / y.shape[0]

def R2(predict_y, test_Y):
    y_mean = np.mean(test_Y)
    _R2 = 1 - np.sum(np.square(predict_y - y_mean)) / np.sum(
        np.square(test_Y - y_mean))
    return _R2

def p1(train_data, test_data):
    print('Problem 1:')
    print('weight', 'bias', 'loss', 'r2', 'feature_name', sep = '\t\t')
    print('-------------------------------------------------------------')
    keys = list(train_data)
    for i in range(len(keys) - 1):
        lm = LinearRegression().fit(
            train_data[keys[i]].values.reshape(len(train_data), 1),
            train_data[keys[8]].values.reshape(len(train_data), 1),
        )
        y = lm.predict(test_data[keys[i]].values.reshape(len(test_data), 1))
        mse = MSE(y, test_data[keys[8]].values.reshape(len(test_data), 1))
        r2 = R2(y, test_data[keys[8]].values.reshape(len(test_data), 1))
        print (
            format(lm.coef_[0][0], '0.6f'),
            format(lm.intercept_[0], '0.6f'),
            format(mse, '0.6f'),
            format(r2, '0.6f'),
            keys[i].split('(')[0],
            sep = '\t'
        )

def SVGD(X, Y, lr, epoch):
    a, b, eps = np.random.randn() * 0.002 - 0.001, np.random.randn() * 0.002 - 0.001, 1e-6
    lr_a, lr_b = eps, eps
    num = len(Y)
    for i in range(epoch):
        y = a * X + b
        da = -sum(X * (Y - y)) / num
        db = -sum(Y - y) / num

        # adagrad : converge faster
        lr_a += da**2
        lr_b += db**2
        a -= lr / np.sqrt(lr_a) * da
        b -= lr / np.sqrt(lr_b) * db
    return a, b



def p2(train_data, test_data):
    print('Problem 2:')
    print('weight', 'bias', 'loss', 'r2', 'feature_name', sep = '\t\t')
    print('-------------------------------------------------------------')
    keys = list(train_data)
    for i in range(len(keys) - 1):
        # train
        a, b = SVGD(
            X = train_data[keys[i]].values.reshape(len(train_data), 1),
            Y = train_data[keys[8]].values.reshape(len(train_data), 1),
            lr = 1,
            epoch = 2000
        )

        # test
        mse = MSE(
            a[0] * test_data[keys[8]].values.reshape(len(test_data), 1) + b[0],
            test_data[keys[8]].values.reshape(len(test_data), 1)
        )

        r2 = R2(
            a[0] * test_data[keys[8]].values.reshape(len(test_data), 1) + b[0],
            test_data[keys[8]].values.reshape(len(test_data), 1)
        )

        print(
            format(a[0], '0.6f'),
            format(b[0], '0.6f'),
            format(mse, '0.6f'),
            format(r2, '0.6f'),
            keys[i].split('(')[0],
            sep = '\t'
        )

def MVGD(X, Y, lr, epoch):
    w = 0.002 * np.random.random_sample(X.shape[1] + 1,) - 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    eps = 1e-8
    X = np.insert(X, 0, 1, axis=1)
    M = np.zeros(len(w))
    V = np.zeros(len(w))
    for i in range(epoch):
        y = np.dot(X, w).reshape(X.shape[0], 1)
        dw = 1 * sum(Y - y) * w

        M = beta_1 * M + (1 - beta_1) * dw
        V = beta_2 * V + (1 - beta_2) * (dw ** 2)

        w -= lr * (M / (1 - beta_1)) / (np.sqrt(V / (1 - beta_2)) + eps)
        if i % 200 == 0:
            print (MSE(w * X, Y))
    return w

def p3(train_data, test_data):
    print('Problem 3:')
    keys = list(train_data)
    lm = LinearRegression().fit(
        train_data[keys[0:8]].values,
        train_data[keys[8]].values.reshape(len(train_data), 1),
    )
    y = lm.predict(test_data[keys[0:8]].values)
    mse = MSE(y, test_data[keys[8]].values.reshape(len(test_data), 1))
    r2 = R2(y, test_data[keys[8]].values.reshape(len(test_data), 1))
    print (
        format(mse, '0.6f'),
        format(r2, '0.6f'),
        sep = '\t'
    )

    print (train_data[[keys[1], keys[3], keys[7]]].values)
    w = MVGD(
        X = train_data[[keys[1], keys[3], keys[7]]].values,
        Y = train_data[keys[8]].values.reshape(len(train_data), 1),
        lr = 0.001,
        epoch = 5000
    )

    mse = MSE(
        w * np.insert(test_data[[keys[1], keys[3], keys[7]]].values, 0, 1, axis=1),
        test_data[keys[8]].values.reshape(len(test_data), 1)
    )

    r2 = R2(
        w * np.insert(test_data[[keys[1], keys[3], keys[7]]].values, 0, 1, axis=1),
        test_data[keys[8]].values.reshape(len(test_data), 1)
    )


    print(
        format(mse, '0.6f'),
        format(r2, '0.6f'),
        sep = '\t'
    )


def main():
    train_data, test_data = load_file('Concrete_Data.csv')
    p1(train_data, test_data)
    # p2(train_data, test_data)
    p3(train_data, test_data)

if __name__ == "__main__":
    main()
