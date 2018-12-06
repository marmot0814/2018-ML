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
    keys = list(df)
    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     sns.scatterplot(df[keys[i]].values.reshape(len(df), ), df[keys[8]].values.reshape(len(df), ))
    #     plt.xlabel(keys[i].split('(')[0])
    #     plt.ylabel(keys[8].split('(')[0])
    # plt.show()

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
        plt.subplot(2, 4, i + 1)
        sns.regplot(
            x = test_data[keys[i]].values.reshape(len(test_data), ),
            y = test_data[keys[8]].values.reshape(len(test_data), )
        )
        plt.xlabel(keys[i].split('(')[0])
        plt.ylabel(keys[8].split('(')[0])
    plt.show()



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
            epoch = 1000
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

        plt.subplot(2, 4, i + 1)
        plt.scatter(test_data[keys[i]].values.reshape(len(test_data), 1), test_data[keys[8]].values.reshape(len(test_data), 1))
        minx = min(test_data[keys[i]].values.reshape(len(test_data), 1))
        maxx = max(test_data[keys[i]].values.reshape(len(test_data), 1))
        plt.plot([minx, maxx], [a[0] * minx + b[0], a[0] * maxx + b[0]])
        plt.xlabel(keys[i].split('(')[0])
        plt.ylabel(keys[8].split('(')[0])
    plt.show()

def MVGD(X, Y, lr, epoch, test_X, test_Y):
    w = 0.002 * np.random.random_sample(X.shape[1] + 1,) - 0.001
    eps = 1e-8
    X = np.insert(X, 0, 1, axis=1)
    test_X = np.insert(test_X, 0, 1, axis = 1)
    G = np.zeros(len(w))
    train_error = []
    test_error = []

    fig = plt.figure()
    plt.ion()
    plt.show()

    error_plot = fig.add_subplot(111)
    for i in range(epoch):
        y = np.dot(X, w).reshape(X.shape[0], 1)
        test_y = np.dot(test_X, w).reshape(test_X.shape[0], 1)
        dw = -1 * sum(Y - y) * w


        if MSE(y, Y) < 300:
            break

        train_err = MSE(y, Y)
        if train_err >= 10000:
            w = 0.002 * np.random.random_sample(X.shape[1],) - 0.001
            G = np.zeros(len(w))

        if i % 500 == 0:
            error_plot.cla()
            print ('epoch = {}'.format(i))
            print ('train')
            print ('---------------------------')
            print('loss', 'r2', sep = '\t\t')
            train_err = MSE(y, Y)
            print(
                format(MSE(y, Y), '0.6f'),
                format(R2(y, Y), '0.6f'),
                sep = '\t'
            )
            print ('test')
            print ('---------------------------')
            print('loss', 'r2', sep = '\t\t')
            test_err = MSE(test_y, test_Y)
            print(
                format(MSE(test_y, test_Y), '0.6f'),
                format(R2(test_y, test_Y), '0.6f'),
                sep = '\t'
            )

            print (w)

            train_err = MSE(y, Y)
            train_error.append(train_err)
            test_error.append(test_err)


            error_plot.plot(list(range(i // 500 + 1)), train_error[0: (i // 500) + 1])
            error_plot.plot(list(range(i // 500 + 1)), test_error[0:(i//500) + 1])
            error_plot.set_xlabel('iteration(* 500 epoch)')
            error_plot.set_ylabel('loss(MSE)')
            error_plot.set_ylim([0, 10000])
            fig.canvas.draw()
        G += dw ** 2
        w -= lr / np.sqrt(G) * dw
    plt.plot()

    return w

def p3(train_data, test_data):
    print('Problem 3:')
    print('loss', 'r2', sep = '\t\t')
    print('-------------------------------------------------------------')
    keys = list(train_data)

    w = MVGD(
        X = train_data[keys[0:8]].values,
        Y = train_data[keys[8]].values.reshape(len(train_data), 1),
        lr = 0.0001,
        epoch = 10000000,
        test_X = test_data[keys[0:8]].values,
        test_Y = test_data[keys[8]].values.reshape(len(test_data), 1)
    )

    mse = MSE(
        w * np.insert(test_data[keys[0:8]].values, 0, 1, axis=1),
        test_data[keys[8]].values.reshape(len(test_data), 1)
    )

    r2 = R2(
        w * np.insert(test_data[keys[0:8]].values, 0, 1, axis=1),
        test_data[keys[8]].values.reshape(len(test_data), 1)
    )


    print(
        format(mse, '0.6f'),
        format(r2, '0.6f'),
        sep = '\t'
    )

def cubic(data):
    cubic_data = data
    cubic_data = np.concatenate((cubic_data, data ** 2), axis=1)
    # cubic_data = np.concatenate((cubic_data, data ** 3), axis=1)
    # cubic_data = np.concatenate((cubic_data, data ** 4), axis=1)
    return cubic_data

def p4(train_data, test_data):
    keys = list(train_data)
    # for i in range(len(keys) - 1):
    i = 5
    X      = cubic(train_data[keys[i]].values.reshape(len(train_data), 1))
    test_X = cubic(test_data[keys[i]].values.reshape(len(test_data), 1))

    w = MVGD(
        X = X,
        Y = train_data[keys[8]].values.reshape(len(train_data), 1),
        lr = 0.0001,
        epoch = 10000000,
        test_X = test_X,
        test_Y = test_data[keys[8]].values.reshape(len(test_data), 1)
    )


def main():
    train_data, test_data = load_file('Concrete_Data.csv')
    # p1(train_data, test_data)
    # p2(train_data, test_data)
    # p3(train_data, test_data)
    p4(train_data, test_data)

if __name__ == "__main__":
    main()
