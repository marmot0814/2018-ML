import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import datetime
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
import random

def load_trend():
    csv = "../data_process/google_trend.csv"
    df = pd.read_csv(csv)[['BTC', 'bitcoin']][::-1]
    return df

def load_coin_mkt():
    csv = "../data_process/coin_market.csv"
    df = pd.read_csv(csv).drop("Date", axis = 1).drop('Unnamed: 0', axis = 1)[::-1]
    return df

def normalized(df):
    return (df - df.mean()) / df.std()

def load_data(k, test_number = 8):
    trend_df = load_trend()
    coin_df = load_coin_mkt()
    money = coin_df['Close**'].values
    trend = normalized(trend_df).values
    coin = normalized(coin_df.drop('Close**', axis = 1)).values
    trend_coin = np.concatenate((trend, coin), axis = 1)

    m = trend_coin.shape[0]
    data = []
    target = []
    for i in range(m - k - 1):
        data.append(trend_coin[i : i + k, :].flatten())
        if money[i + k] >= money[i + k + 1]: # 跌
            target.append([0, 1])
        else:                               # 漲
            target.append([1, 0])
    data = np.array(data)
    target = np.array(target)

    train_data = data[:-test_number, :]
    train_target = target[:-test_number, :]

    test_data = data[-test_number:, :]
    test_target = target[-test_number:, :]
    test_money = money[-test_number - 1:]

    return train_data, train_target, test_data, test_target, test_money

def go(model, test_data, test_money, coin = 10000):
    state = 0   # 0: 沒買, 1: 有買
    seld = 0
    for i in range(test_data.shape[0]):
        output = model.predict(test_data[i, :].reshape(1, test_data.shape[1]))
        res = output.argmax()
        if res == state:
            if state == 0:
                seld = coin / test_money[i]
                state = 1
                coin = 0
            else:
                coin = seld * test_money[i]
                state = 0
                seld = 0
    return coin + seld * test_money[-1]

k = 48
train_data, train_target, test_data, test_target, test_money = load_data(k, 24)

model = Sequential()
model.add(Dense(input_dim = 7 * k, units = 1000, activation='relu'))
model.add(Dense(units = 1000, activation = 'relu'))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
max_output = -1
epochs = 200
while epochs:
    epochs -= 1
    model.fit(train_data, train_target, batch_size = 100, epochs=1)
    output = go(model, test_data, test_money)
    if output > max_output:
        max_output = output
        model.save('best.h5')

model = load_model('best.h5')
print ("if you throw 10000 coin in the market, you will get {0} coin in reward.".format(go(model, test_data, test_money)))
print ("train acc: {0}".format(model.evaluate(train_data, train_target)[1]))
print ("test acc: {0}".format(model.evaluate(test_data, test_target)[1]))
