import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import csv
import random

MPa = 'Concrete compressive strength(MPa, megapascals) '


def load_file(filename):
    with open(filename, newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))
        datas = [list(i.values()) for i in reader]
        keys = list(reader[0].keys())

        #shuffle
        np.random.shuffle(datas)

        train_datas = datas[0:int(0.8 * len(datas))]
        test_datas = datas[int(0.8 * len(datas)):]

        return train_datas, test_datas, keys, keys.index(MPa)


def gradient_descent(X, Y, lr, epoch):
    a, b = 0.0, 0.0
    lr_a, lr_b = 1e-6, 1e-6
    num = len(Y)
    for i in range(epoch):
        y = a * X + b
        cost = sum([i**2 for i in (Y - y)]) / num
        da = -(2 / num) * sum(X * (Y - y))
        db = -(2 / num) * sum(Y - y)
        # softmax # overflow
        da = max(da, -1) if da < 0 else min(da, 1)
        db = max(db, -1) if db < 0 else min(db, 1)

        # adagrad : converge faster
        lr_a += da**2
        lr_b += da**2
        a -= lr / np.sqrt(lr_a) * da
        b -= lr / np.sqrt(lr_b) * db
    return a, b


def plot(X, Y, weight, bias, keys):
    frames = len(weight)
    X = np.array(X).astype(np.float)
    Y = np.array(Y).astype(np.float)
    weight = np.array(weight).astype(np.float)
    bias = np.array(bias).astype(np.float)

    min_X = np.amin(X, axis=0)
    max_X = np.amax(X, axis=0)

    for frame in range(frames):
        plt.subplot(2, 4, frame + 1)

        minx = min_X[frame]
        maxx = max_X[frame]

        if minx < 0:  # expand line length
            minx *= 1.1
        else:
            minx *= 0.9

        if maxx > 0:
            maxx *= 1.1
        else:
            maxx *= 0.9

        plt.plot([minx, maxx],
                 [minx * weight[frame] + bias, maxx * weight[frame] + bias])

        plt.scatter(np.reshape(X[:, frame], (1, len(X))), Y, s=5)

        plt.xlabel(keys[frame].split('(')[0])
        plt.tight_layout()
    plt.show()


def p1(datas, test_datas, keys, index):
    datas = np.array(datas).astype(np.float)  #training data
    train_data = datas[:, 0:len(keys) - 1]
    """
    const_one_arr = np.ones((len(train_data), 1))
    train_data = np.append(
        const_one_arr, train_data, axis=1)  #append const column
    """
    target_data = datas[:, index]  #training target data
    lm = LinearRegression().fit(train_data, target_data)
    test_datas = np.array(test_datas).astype(np.float)  #testing data
    test_data = test_datas[:, 0:len(keys) - 1]
    """
    const_one_arr = np.ones((len(test_datas), 1))
    test_data = np.append(const_one_arr, test_data, axis=1)
    """
    test_target_data = test_datas[:, index]  #testing target data

    print("r2_score: ", lm.score(test_data, test_target_data))
    print("weight: ", lm.coef_[0:])
    print("bias: ", lm.intercept_)

    plot(test_data[:, 1:], test_target_data, lm.coef_[1:], lm.coef_[0], keys)


def p2(datas, keys, index, epoch):
    for i in range(len(keys)):
        # for i in range(1):
        data = np.array(datas)[:, i].astype(np.float)
        pidt = np.array(datas)[:, index].astype(np.float)
        a, b = gradient_descent(data, pidt, 0.1, epoch)
        print(a, b, keys[i])


def MVGD(X, Y, lr, epoch):  # multi-variable gradient descent
    # init
    w = (-0.2) + 0.4 * np.random.random_sample(X.shape[1],
                                              )  # sample from -0.2 ~ 0.2
    # w0 = 1, xi0 = 1
    w = np.insert(w, 0, 1)
    X = np.insert(X, 0, 1, axis=1)
    G = np.ones(len(w)) * 1e-6  # adagrad
    num = X.shape[0]
    # y = wx + b
    for _epoch in range(epoch):
        y = np.dot(w, X.T)
        dw = -1 * np.array(
            [(np.sum([X[i][j] * (Y[i] - np.dot(w, X[i]))
                      for i in range(num)])) / num
             for j in range(len(w))])
        G += dw**2
        w -= lr * (dw / np.sqrt(G))
    return w


def p3(datas, keys, y_index, epoch, test_datas):
    #print(datas[0])
    datas = np.array(datas).astype(np.float)  #training data
    lr = 0.5
    w = MVGD(datas[:, 0:y_index], datas[:, y_index], lr, epoch)
    print("w = {}".format(w))

    # MSE
    test_datas = np.array(test_datas).astype(np.float)
    test_Y = test_datas[:, y_index]
    test_X = np.insert(test_datas[:, 0:y_index], 0, 1, axis=1)
    y = np.dot(w, test_X.T)
    MSE = np.sum(np.square(test_Y - y)) / test_X.shape[0]
    print("MSE = {}".format(MSE))


def main():
    train_datas, test_datas, keys, index = load_file('Concrete_Data.csv')

    #print('Problem 1:')
    #p1(train_datas, test_datas, keys, index)
    #print('=============================')
    epoch = 3000
    #print('Problem 2:')
    #p2(train_datas, keys, index, epoch)
    print('=============================')
    print('Problem 3:')
    p3(train_datas, keys, index, epoch, test_datas)

    print('=============================')


if __name__ == "__main__":
    main()
