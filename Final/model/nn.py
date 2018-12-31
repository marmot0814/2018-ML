import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils

def load_trend():
    csv = "../data_process/google_trend.csv"
    df = pd.read_csv(csv)[['BTC', 'bitcoin']][::-1]
    df = (df - df.mean()) / df.std()
    return df


def load_coin_mkt():
    csv = "../data_process/coin_market.csv"
    df = pd.read_csv(csv).drop("Date", axis = 1).drop('Unnamed: 0', axis = 1)[::-1]
    df["Market Cap"] = df["Market Cap"].str.replace(',', '').astype(float)
    df["Volume"] = df["Volume"].str.replace(',', '').astype(float)
    df = (df - df.mean()) / df.std()
    print (df.head())
    return df

def load_data(k, ratio = 0.7):
    trend = load_trend().values
    coin = load_coin_mkt().values
    m = trend.shape[0]
    data_ = np.concatenate((trend, coin), axis=1)
    TARGET_INDEX = [5]
    DATA_INDEX = [i for i in range(np.shape(data_)[1]) if i not in TARGET_INDEX]
    data = []
    target = []
    for i in range(m - k - 1):
        data.append(data_[i:i + k, DATA_INDEX].flatten())
        if data_[i+k, TARGET_INDEX] > data_[i+k-1, TARGET_INDEX]:
            target.append([1, 0])
        else:
            target.append([0, 1])
    ordered_data = np.array(data)
    ordered_target = np.array(target)

    # shuffle
    idx = np.arange(m - k - 1)
    np.random.shuffle(idx)
    data = ordered_data[idx]
    target = ordered_target[idx]

    x = round((m - k - 1) * ratio)
    print ("train: {0}, test: {1}".format(x, m - k - 1 - x))
    train_data = data[:x, :]
    train_target = target[:x, :]
    test_data = data[x:, :]
    test_target = target[x:, :]
    return ordered_data, ordered_target, train_data, train_target, test_data, test_target

def test(model, ordered_data, k, coin = 1):
    state = 0   # 0: 沒買, 1: 有買
    csv = "../data_process/coin_market.csv"
    money = pd.read_csv(csv)[::-1]['Close**'].values
    m = ordered_data.shape[0]
    seld = 0
    for i in range(m):
        output = model.predict(ordered_data[i, :].reshape(1, ordered_data.shape[1]))
        res = output.argmax()
        if res == state:
            if state == 0:
                seld = coin / money[i + k - 1]
                state = 1
                coin = 0
            else:
                coin = seld * money[i + k - 1]
                state = 0
                seld = 0
    return coin + seld * money[-1]


k = 48 * 2
ordered_data, ordered_target, train_data, train_target, test_data, test_target = load_data(k, 0.7)

model = Sequential()
model.add(Dense(input_dim = 7 * k, units = 1000, activation='relu'))
model.add(Dropout(0.7122))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_target, batch_size = 100, epochs=1000)

res = model.evaluate(test_data, test_target)

print ("test acc = {0}".format(res[1]))

coin = test(model, ordered_data, k)
print ("if you throw 1 coin, you will get {0} coin in reward.".format(coin))

