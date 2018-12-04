import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import csv
import random
import pandas as pd
import seaborn as sns
import itertools
sns.set()
MPa = 'Concrete compressive strength(MPa, megapascals) '

def normarlize(data):
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    return data

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
    return a, b, cost

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

def plot_p1(data,pidt,lm):
    plt.scatter(data, pidt, color='black')
    plt.plot(data, lm.predict(np.reshape(data, (len(data), 1))), color='blue', linewidth=3)
    # plt.plot(to_be_predicted, predicted_sales, color = 'red', marker = '^', markersize = 10)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

def p1(datas, test_datas, keys, index):
    print('Problem 1:')
    print('weight  ','bias    ','loss    ','feature_name')
    print('-----------------------------')
    for i in range(len(keys)):
        data = np.array(datas)[:,i].astype(np.float)
        target = np.array(datas)[:,index].astype(np.float)
        lm = LinearRegression()
        lm.fit(np.reshape(data,(len(data),1)),np.reshape(target,(len(target),1)))

        test_data = np.array(test_datas)[:,i].astype(np.float)
        test_target = np.array(test_datas)[:,index].astype(np.float)
        # plot(test_data, test_pidt, lm.coef_, lm.coef_, keys[i])
        # plot_p1(test_data,test_pidt,lm)
        a, b = lm.coef_[0][0],lm.intercept_[0]
        test_pidt = a * test_data + b
        cost = sum([i**2 for i in (test_target - test_pidt)]) / len(test_target)

        print(format(a,'0.6f'),format(b,'0.6f'),format(cost,'0.6f'),keys[i].split('(')[0]) # 印出係數 截距

def p2(datas, keys, index, epoch):
    print('=============================')
    print('Problem 2:')
    print('-----------------------------')
    for i in range(len(keys)):
        # for i in range(1):
        data = np.array(datas)[:, i].astype(np.float)
        pidt = np.array(datas)[:, index].astype(np.float)
        a, b, loss = gradient_descent(data, pidt, 0.1, epoch)
        print(format(a,'0.6f'),format(b,'0.6f'),format(loss,'0.6f'),keys[i].split('(')[0]) # 印出係數 截距

def MVGD(X, Y, lr, epoch):  # multi-variable gradient descent
    # init
    w = (-0.2) + 0.4 * np.random.random_sample(X.shape[1] + 1,
                                              )  # sample from -0.2 ~ 0.2
    # w0 = 1, xi0 = 1
    #w = np.insert(w, 0, 1)
    X = np.insert(X, 0, 1, axis=1)
    G = np.ones(len(w)) * 1e-6  # adagrad
    num = X.shape[0]
    #print(X.shape)
    # y = wx + b
    for _epoch in range(epoch):
        if _epoch % 10 == 0:
            print(w)
        y = np.dot(w, X.T)
        dw = -1 * np.array(
            [(np.sum([X[i][j] * (Y[i] - np.dot(w, X[i]))
                      for i in range(num)])) / num
             for j in range(len(w))])
        G += dw**2
        G = 1
        w -= lr * (dw / np.sqrt(G))

    return w

def MSE(w, test_X, test_Y):
    y = np.dot(w, test_X.T)
    MSE = np.sum(np.square(test_Y - y)) / test_X.shape[0]
    return MSE

def p3(datas, keys, y_index, epoch, test_datas, lr):
    print('=============================')
    print('Problem 3:')
    print('-----------------------------')
    #print(datas[0])
    datas = np.array(datas).astype(np.float)  #training data
    w = MVGD(datas[:, 0:y_index], datas[:, y_index], lr, epoch)
    print("w = {}".format(w))

    # MSE
    test_datas = np.array(test_datas).astype(np.float)
    test_Y = test_datas[:, y_index]
    test_X = test_datas[:, 0:y_index]
    test_X = np.insert(test_X, 0, 1, axis=1)  # insert x0
    _MSE = MSE(w, test_X, test_Y)
    print("MSE = {}".format(_MSE))


def cubicGD(datas, data_indexs, Y, epoch, lr):  # 三次
    # init
    datas = datas[:, data_indexs]
    #print(datas)
    new_datas = []
    for _data in datas:
        new_data = []
        _data = np.append(_data, 1)
        for indexs in itertools.combinations_with_replacement(
                range(len(data_indexs) + 1), 3):
            #print(indexs)
            new_data.append(np.product([_data[index] for index in indexs]))

        new_data.pop()
        new_datas.append(new_data)
    new_datas = np.array(new_datas)
    w = MVGD(new_datas, Y, lr, epoch)

    train_MSE = MSE(w, np.insert(new_datas, 0, 1, axis=1), Y)
    print("train_MSE = {}".format(train_MSE))

    return w

def p4(datas, data_indexs, keys, y_index, epoch, test_datas, lr):
    datas = np.array(datas).astype(np.float)  #training data
    datas = normarlize(datas)
    w = cubicGD(datas, data_indexs, datas[:, y_index], epoch, lr)
    print("w = {}".format(w))
    # MSE
    #print(epoch)
    test_datas = np.array(test_datas).astype(np.float)
    test_datas = normarlize(test_datas)
    test_Y = test_datas[:, y_index]
    test_X = test_datas[:, data_indexs]
    tmp_test_X = []
    for row in test_X:
        new_row = [1]  # insert xi0 = 1
        row = np.append(row, 1)
        for indexs in itertools.combinations_with_replacement(
                range(row.shape[0]), 3):
            #print(indexs)
            new_row.append(np.product([row[index] for index in indexs]))

        new_row.pop()
        tmp_test_X.append(new_row)

    test_X = np.array(tmp_test_X)
    print(test_X.shape[1])
    _MSE = MSE(w, test_X, test_Y)
    print("MSE = {}".format(_MSE))

def plot_unity(xdata, ydata, **kwargs):
    sns.regplot(
        xdata,
        ydata,
        scatter_kws={
            's': 2,
            "color": "black"
        },
        line_kws={"color": "red"})

def pairplot():
    df = pd.read_csv('Concrete_Data.csv')
    g = sns.PairGrid(df, height=5)
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plot_unity)
    for i in range(9):
        for j in range(9):
            g.axes[i, j].set_xlabel(
                g.axes[i, j].get_xlabel(), rotation=20, size=7)
            g.axes[i, j].set_ylabel('')
    plt.subplots_adjust(bottom=0.15, left=0.05, top=0.95, right=0.95)
    plt.show()

def main():
    train_datas, test_datas, keys, index = load_file('Concrete_Data.csv')
    epoch = 1000
    p1(train_datas, test_datas, keys, index)
    p2(train_datas, keys, index, epoch)
    #pairplot()
    #p3(train_datas, keys, index, epoch, test_datas, lr=0.5)
    #print('Problem 4:')
    #for i in range(len(train_datas[0])):
    #datas = []
    #for i in range(1000):
    #    datas.append([i, i**3])
    #p4(train_datas, [2, 4, 8], keys, index, epoch, test_datas, lr=0.01)
    #p4(datas, [0], keys, 1, epoch, datas, lr=0.01)
    #print('=============================')


if __name__ == "__main__":
    main()
