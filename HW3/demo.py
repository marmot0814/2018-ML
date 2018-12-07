import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import time


def load_file(filename, visualized=True):
    df = pd.read_csv(filename, dtype=np.float128)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    keys = list(df)
    if visualized:
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            sns.scatterplot(df[keys[i]].values.reshape(len(df),),
                            df[keys[8]].values.reshape(len(df),))
            plt.xlabel(keys[i].split('(')[0], fontSize=10)
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


def LogCosh(y, Y):
    return np.sum(np.log(np.cosh(y - Y))) / y.shape[0]


def R2(y, Y):
    return 1 - np.sum(np.square(Y - y)) / np.sum(np.square(Y - Y.mean()))


def p1(train_data, test_data):
    print('Problem 1:')
    print('weight', 'bias', 'loss', 'r2', 'feature_name', sep='\t\t')
    print('-------------------------------------------------------------')
    train_data = train_data.astype(dtype=np.float16)
    test_data = test_data.astype(dtype=np.float16)

    keys = list(train_data)
    for i in range(len(keys) - 1):
        lm = LinearRegression().fit(
            train_data[keys[i]].values.reshape(len(train_data), 1),
            train_data[keys[8]].values.reshape(len(train_data), 1),
        )
        y = lm.predict(test_data[keys[i]].values.reshape(len(test_data), 1))
        mse = MSE(y, test_data[keys[8]].values.reshape(len(test_data), 1))
        r2 = R2(y, test_data[keys[8]].values.reshape(len(test_data), 1))
        print(
            format(lm.coef_[0][0], '0.6f'),
            format(lm.intercept_[0], '0.6f'),
            format(mse, '0.6f'),
            format(r2, '0.6f'),
            keys[i].split('(')[0],
            sep='\t')
        plt.subplot(2, 4, i + 1)
        sns.regplot(
            x=test_data[keys[i]].values.reshape(len(test_data)),
            y=test_data[keys[8]].values.reshape(len(test_data)))
        plt.xlabel(keys[i].split('(')[0])
        plt.ylabel(keys[8].split('(')[0])
        plt.axis("equal")
    plt.show()


def SVGD(X, Y, lr, epoch):
    a, b, eps = np.random.randn() * 0.002 - 0.001, np.random.randn(
    ) * 0.002 - 0.001, 1e-8
    M_a, M_b = 0, 0
    V_a, V_b = 0, 0
    beta_1, beta_2 = 0.9, 0.999
    error = np.inf
    cnt = 0
    num = len(Y)
    while True:
        y = a * X + b
        da = -sum(X * (Y - y)) / num
        db = -sum(Y - y) / num

        # adam : converge faster
        M_a = beta_1 * M_a + (1 - beta_1) * da
        M_b = beta_1 * M_b + (1 - beta_1) * db
        V_a = beta_2 * V_a + (1 - beta_2) * (da**2)
        V_b = beta_2 * V_b + (1 - beta_2) * (db**2)
        a -= lr * M_a / (1 - beta_1) / np.sqrt(V_a / (1 - beta_2) + eps)
        b -= lr * M_b / (1 - beta_1) / np.sqrt(V_b / (1 - beta_2) + eps)

        if abs(MSE(y, Y) - error) < 0.001:
            cnt += 1
            if cnt > 100:
                break
        else:
            cnt = 0

        error = MSE(y, Y)
    return a, b


def p2(train_data, test_data):
    print('Problem 2:')
    print('weight', 'bias', 'loss', 'r2', 'feature_name', sep='\t\t')
    print('-------------------------------------------------------------')
    keys = list(train_data)

    for i in range(len(keys) - 1):
        X = train_data[keys[i]].values.reshape(len(train_data), 1)
        Y = train_data[keys[8]].values.reshape(len(train_data), 1)
        test_X = test_data[keys[i]].values.reshape(len(test_data), 1)
        test_Y = test_data[keys[8]].values.reshape(len(test_data), 1)
        # train
        a, b = SVGD(X=X, Y=Y, lr=0.5, epoch=5000)

        # test
        mse = MSE(a[0] * test_X + b[0], test_Y)
        r2 = R2(a[0] * test_X + b[0], test_Y)

        print(
            format(a[0], '0.6f'),
            format(b[0], '0.6f'),
            format(mse, '0.6f'),
            format(r2, '0.6f'),
            keys[i].split('(')[0],
            sep='\t')

        plt.subplot(2, 4, i + 1)
        plt.scatter(test_X, test_Y)
        minx = min(test_X)
        maxx = max(test_X)
        plt.plot([minx, maxx], [a[0] * minx + b[0], a[0] * maxx + b[0]], 'r')
        plt.xlabel(keys[i].split('(')[0])
        plt.ylabel(keys[8].split('(')[0])
        plt.axis("equal")
    plt.show()


def MVGD(X,
         Y,
         lr,
         epoch,
         test_X,
         test_Y,
         optimizer,
         error_min,
         error_max,
         pids,
         lossFunction='MSE'):
    w = 0.002 * np.random.random_sample(X.shape[1]) - 0.001
    eps = 1e-8

    if optimizer == 'adagrad':
        G = np.zeros(len(w))

    if optimizer == 'adam':
        M = np.zeros(len(w))
        V = np.zeros(len(w))
        beta_1 = 0.9
        beta_2 = 0.999

    train_error = []
    test_error = []
    train_R2 = []
    test_R2 = []

    plt.ion()
    plt.show()
    error_fig = plt.figure()
    error_plot = error_fig.add_subplot(111)
    R2_fig = plt.figure()
    R2_plot = R2_fig.add_subplot(111)

    output_density = 100

    for i in range(epoch):

        y = np.dot(X, w).reshape(X.shape[0], 1)
        test_y = np.dot(test_X, w).reshape(test_X.shape[0], 1)

        if lossFunction == 'MSE':
            dw = -1 * (Y - y).T.dot(X).reshape(X.shape[1]) / X.shape[0]
        if lossFunction == 'LogCosh':
            dw = -((np.sinh(Y - y) / np.cosh(Y - y)).T.dot(X)).reshape(
                X.shape[1]) / X.shape[0]

        if i % output_density == 0:
            error_plot.cla()

            if lossFunction == 'MSE':
                train_err = MSE(y, Y)
                test_err = MSE(test_y, test_Y)

            if lossFunction == 'LogCosh':
                train_err = LogCosh(y, Y)
                test_err = LogCosh(test_y, test_Y)

            train_error.append(train_err)
            test_error.append(test_err)

            train_r2 = R2(y, Y)
            test_r2 = R2(test_y, test_Y)

            train_R2.append(train_r2)
            test_R2.append(test_r2)

            error_plot.set_title(
                'Problem {}, train_error = {:0.3f}, test_error = {:0.3f}'.
                format(pids, train_err, test_err))
            error_plot.plot(
                list(range(i // output_density + 1)),
                train_error[0:(i // output_density) + 1], 'blue')
            error_plot.plot(
                list(range(i // output_density + 1)),
                test_error[0:(i // output_density) + 1], 'orange')
            error_plot.set_xlabel(
                'iteration(* {} epoch)'.format(output_density))

            R2_plot.set_title(
                'Problem {}, train_r2 = {:0.3f}, test_r2 = {:0.3f}'.format(
                    pids, train_r2, test_r2))
            R2_plot.plot(
                list(range(i // output_density + 1)),
                train_R2[0:(i // output_density) + 1],
                'blue',
            )
            R2_plot.plot(
                list(range(i // output_density + 1)),
                test_R2[0:(i // output_density) + 1], 'orange')
            R2_plot.set_xlabel('iteration(* {} epoch)'.format(output_density))

            if lossFunction == 'MSE':
                error_plot.set_ylabel('loss(MSE)')
                R2_plot.set_ylabel('loss(MSE)')

            if lossFunction == 'LogCosh':
                error_plot.set_ylabel('loss(LogCosh)')
                R2_plot.set_ylabel('loss(LogCosh)')

            #plt.legend(labels=["line1", "line2"], loc='upper right')
            error_plot.set_ylim([0, error_max])
            error_plot.legend(labels=['train_data error', 'test_data error'])
            R2_plot.set_ylim([0, 1])
            R2_plot.legend(labels=['train_data R2', 'test_data R2'])
            error_fig.canvas.draw()
            R2_fig.canvas.draw()
            # plt.pause(0.01)

        if lossFunction == 'MSE':
            train_err = MSE(y, Y)

        if lossFunction == 'LogCosh':
            train_err = LogCosh(y, Y)

        # print ('test_err({}) = {}'.format(lossFunction, test_err))
        # print ('train_err({}) = {}'.format(lossFunction, train_err))
        # print ('R2 = {}'.format(R2(test_y, test_Y)))

        if optimizer == 'adagrad':
            G += dw**2
            w -= lr / np.sqrt(G) * dw

        if optimizer == 'adam':
            M = beta_1 * M + (1 - beta_1) * dw
            V = beta_2 * V + (1 - beta_2) * (dw**2)
            w -= lr * M / (1 - beta_1) / (np.sqrt(V / (1 - beta_2)) + eps)

    return w


def p3(train_data, test_data):
    print('Problem 3:')
    keys = list(train_data)
    X = Linear(train_data[keys[0:8]].values)
    Y = train_data[keys[8]].values.reshape(len(train_data), 1)
    test_X = Linear(test_data[keys[0:8]].values)
    test_Y = test_data[keys[8]].values.reshape(len(test_data), 1)

    w = MVGD(
        X=X,
        Y=Y,
        lr=0.0001,
        epoch=20000,
        optimizer='adam',
        test_X=test_X,
        test_Y=test_Y,
        error_min=0,
        error_max=1000,
        pids=3)

    mse = MSE(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)
    r2 = R2(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)

    print('loss(MSE)', 'r2', sep='\t')
    print('-------------------------------------------------------------')
    print(format(mse, '0.6f'), format(r2, '0.6f'), sep='\t')


def Linear(data):
    res_data = np.array([[1] for i in range(data.shape[0])])
    res_data = np.concatenate((res_data, data), axis=1)
    return res_data


def Squared(data):
    res_data = np.array([[1] for i in range(data.shape[0])])
    data = np.concatenate((data, res_data), axis=1)
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            res_data = np.concatenate((res_data, data[:, i].reshape(
                data.shape[0], 1) * data[:, j].reshape(data.shape[0], 1)),
                                      axis=1)
    return res_data[:, 1:]


def Cubic(data):
    res_data = np.array([[1] for i in range(data.shape[0])])
    data = np.concatenate((data, res_data), axis=1)
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            for k in range(j, data.shape[1]):
                res_data = np.concatenate(
                    (res_data, data[:, i].reshape(data.shape[0], 1) *
                     data[:, j].reshape(data.shape[0], 1) * data[:, k].reshape(
                         data.shape[0], 1)),
                    axis=1)
    return res_data[:, 1:]


def p4(train_data, test_data):
    print('Problem 4:')
    keys = list(train_data)
    X = Squared(train_data[keys[0:8]].values)
    Y = train_data[keys[8]].values.reshape(len(train_data), 1)
    test_X = Squared(test_data[keys[0:8]].values)
    test_Y = test_data[keys[8]].values.reshape(len(test_data), 1)

    w = MVGD(
        X=X,
        Y=train_data[keys[8]].values.reshape(len(train_data), 1),
        lr=0.00001,
        optimizer='adam',
        epoch=20000,
        test_X=test_X,
        test_Y=test_data[keys[8]].values.reshape(len(test_data), 1),
        error_min=0,
        error_max=1000,
        pids=4)

    mse = MSE(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)
    r2 = R2(np.dot(test_X, w).reshape(test_X.shape[0], 1), test_Y)

    print('loss(MSE)', 'r2', sep='\t')
    print('-------------------------------------------------------------')
    print(format(mse, '0.6f'), format(r2, '0.6f'), sep='\t')

def main():
    train_data, test_data = load_file('Concrete_Data.csv', visualized=False)
    # p1(train_data, test_data)
    # p2(train_data, test_data)
    p3(train_data, test_data)
    p4(train_data, test_data)


if __name__ == "__main__":
    main()
