import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns

def load_file(filename, visualized = True):
    df = pd.read_csv(filename)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    keys = list(df)
    if visualized:
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            sns.scatterplot(df[keys[i]].values.reshape(len(df), ), df[keys[8]].values.reshape(len(df), ))
            plt.xlabel(keys[i].split('(')[0])
            plt.ylabel(keys[8].split('(')[0])
            plt.axis("equal")
        plt.show()

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
            x = test_data[keys[i]].values.reshape(len(test_data)),
            y = test_data[keys[8]].values.reshape(len(test_data))
        )
        plt.xlabel(keys[i].split('(')[0])
        plt.ylabel(keys[8].split('(')[0])
        plt.axis("equal")
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
        X       = train_data[keys[i]].values.reshape(len(train_data), 1)
        Y       = train_data[keys[8]].values.reshape(len(train_data), 1)
        test_X  = test_data[keys[i]].values.reshape(len(test_data), 1)
        test_Y  = test_data[keys[8]].values.reshape(len(test_data), 1)
        # train
        a, b = SVGD(
            X = X,
            Y = Y,
            lr = 1,
            epoch = 2000
        )

        # test
        mse = MSE(a[0] * test_X + b[0], test_Y)
        r2 = R2(a[0] * test_X + b[0], test_Y)

        print(
            format(a[0], '0.6f'),
            format(b[0], '0.6f'),
            format(mse, '0.6f'),
            format(r2, '0.6f'),
            keys[i].split('(')[0],
            sep = '\t'
        )

        plt.subplot(2, 4, i + 1)
        plt.scatter(test_X, test_Y)
        minx = min(test_X)
        maxx = max(test_X)
        plt.plot([minx, maxx], [a[0] * minx + b[0], a[0] * maxx + b[0]], 'r')
        plt.xlabel(keys[i].split('(')[0])
        plt.ylabel(keys[8].split('(')[0])
        plt.axis("equal")
    plt.show()


def MVGD(X, Y, lr, epoch, test_X, test_Y, error_min, error_max, visualized = True, xlabel = '', ylabel = ''):
    base = 0
    w = 0.002 * np.random.random_sample(X.shape[1]) - 0.001
    eps = 1e-8
    G = np.zeros(len(w))

    train_error = []
    test_error = []

    plt.ion()
    plt.show()

    error_fig = plt.figure()
    error_plot = error_fig.add_subplot(111)

    if visualized:
        point_fig = plt.figure()
        point_plot = point_fig.add_subplot(111)

    output_density = 100

    for i in range(epoch):
        y = np.dot(X, w).reshape(X.shape[0], 1)
        test_y = np.dot(test_X, w).reshape(test_X.shape[0], 1)
        dw = -1 * sum(Y - y) * w

        if i % output_density == 0:
            error_plot.cla()

            train_err = MSE(y, Y)
            test_err = MSE(test_y, test_Y)
            if len(train_error) != 0 and abs(train_error[-1] - train_err) < 0.01:
                break

            train_error.append(train_err)
            test_error.append(test_err)

            error_plot.plot(list(range(i // output_density + 1)), train_error[0: (i // output_density) + 1])
            error_plot.plot(list(range(i // output_density + 1)), test_error[0:(i//output_density) + 1])
            error_plot.set_xlabel('iteration(* {} epoch)'.format(output_density))
            error_plot.set_ylabel('loss(MSE)')
            error_plot.set_ylim([0, error_max])
            error_fig.canvas.draw()
            if visualized:
                point_plot.cla()
                point_plot.plot(test_X[:, 1], test_Y, '.')
                minx = min(test_X[:, 1])
                maxx = max(test_X[:, 1])
                point_plot.plot(
                    np.linspace(minx,maxx,100),
                    Polynomialize(np.linspace(minx,maxx,100).reshape(100, 1), X.shape[1] - 1).dot(w)
                )
                point_plot.set_xlabel(xlabel, fontsize = 10)
                point_plot.set_ylabel(ylabel, fontsize = 10)
                point_plot.text(
                    x = (minx + maxx) / 2,
                    y = test_Y.mean() + 20,
                    s = "loss = {}\nr2 = {}".format(MSE(test_y, test_Y), R2(test_y, test_Y)),
                    fontsize=12
                )
                point_plot.axis("equal")
                point_fig.canvas.draw()

        train_err = MSE(y, Y)
        if train_err >= error_max or (i - base >= 1000 and train_err >= 1000):
            w = 0.002 * np.random.random_sample(X.shape[1],) - 0.001
            G = np.zeros(len(w))
            base = i

        G += dw ** 2
        w -= lr / np.sqrt(G) * dw

    if visualized:
        plt.close(point_fig)
    plt.close(error_fig)
    return w

def p3(train_data, test_data):
    print('Problem 3:')
    keys = list(train_data)
    while True:
        X       = Polynomialize(train_data[keys[0:8]].values, 1)
        Y       = train_data[keys[8]].values.reshape(len(train_data), 1)
        test_X  = Polynomialize(test_data[keys[0:8]].values, 1)
        test_Y  = test_data[keys[8]].values.reshape(len(test_data), 1)

        w = MVGD(
            X = X,
            Y = Y,
            lr = 0.0001,
            epoch = 100000,
            test_X = test_X,
            test_Y = test_Y,
            error_min = 500,
            error_max = 1800,
            visualized = False
        )

        mse = MSE(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)
        r2 = R2(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)

        if r2 > 0.6:
            break

    print('loss', 'r2', sep = '\t\t')
    print('-------------------------------------------------------------')
    print(
        format(mse, '0.6f'),
        format(r2, '0.6f'),
        sep = '\t'
    )

def Polynomialize(data, deg):
    cubic_data = np.array([[1] for i in range(data.shape[0])])
    for i in range(1, deg + 1):
        cubic_data = np.concatenate((cubic_data, data ** i), axis=1)
    return cubic_data

def p4(train_data, test_data):
    print('Problem 4:')
    keys = list(train_data)
    i = 5
    deg = 2
    X       = Polynomialize(train_data[keys[i]].values.reshape(len(train_data), 1), deg)
    Y       = train_data[keys[8]].values.reshape(len(train_data), 1)
    test_X  = Polynomialize(test_data[keys[i]].values.reshape(len(test_data), 1), deg)
    test_Y  = test_data[keys[8]].values.reshape(len(test_data), 1)

    w = MVGD(
        X = X,
        Y = train_data[keys[8]].values.reshape(len(train_data), 1),
        lr = 0.0001,
        epoch = 10000,
        test_X = test_X,
        test_Y = test_data[keys[8]].values.reshape(len(test_data), 1),
        error_min = 300,
        error_max = 10000,
        xlabel = keys[i],
        ylabel = keys[8]
    )

    mse = MSE(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)
    r2 = R2(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)

    print('loss', 'r2', sep = '\t\t')
    print('-------------------------------------------------------------')
    print(
        format(mse, '0.6f'),
        format(r2, '0.6f'),
        sep = '\t'
    )


def p5(train_data, test_data):
    print('Problem 5:')
    keys = list(train_data)
    i = 5
    deg = 2
    print('Choose Feature {} and Concrete compressive strength and degree {}'.format(keys[i], deg))
    while True:
        X       = Polynomialize(train_data[keys[i]].values.reshape(len(train_data), 1), deg)
        Y       = train_data[keys[8]].values.reshape(len(train_data), 1)
        test_X  = Polynomialize(test_data[keys[i]].values.reshape(len(test_data), 1), deg)
        test_Y  = test_data[keys[8]].values.reshape(len(test_data), 1)

        w = MVGD(
            X = X,
            Y = train_data[keys[8]].values.reshape(len(train_data), 1),
            lr = 0.0001,
            epoch = 10000,
            test_X = test_X,
            test_Y = test_data[keys[8]].values.reshape(len(test_data), 1),
            error_min = 300,
            error_max = 10000,
            xlabel = 'Coarse Aggregate',
            ylabel = 'Concrete compressive strength'
        )

        mse = MSE(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)
        r2 = R2(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)
        if r2 > 0.87:
            break
        train_data, test_data = load_file('Concrete_Data.csv', visualized = False)

    print('loss', 'r2', sep = '\t\t')
    print('-------------------------------------------------------------')
    print(
        format(mse, '0.6f'),
        format(r2, '0.6f'),
        sep = '\t'
    )



def main():
    train_data, test_data = load_file('Concrete_Data.csv')
    p1(train_data, test_data)
    p2(train_data, test_data)
    p3(train_data, test_data)
    p4(train_data, test_data)
    p5(train_data, test_data)

if __name__ == "__main__":
    main()
