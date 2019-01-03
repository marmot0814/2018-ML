import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import datetime
from sklearn.svm import SVC


def load_trend():
    csv = "../data_process/google_trend.csv"
    df = pd.read_csv(csv)[['BTC', 'bitcoin']][::-1]
    return df


def load_coin_mkt():
    csv = "../data_process/coin_market.csv"
    df = pd.read_csv(csv).drop("Date", axis = 1).drop('Unnamed: 0', axis = 1)[::-1]
    df["Market Cap"] = df["Market Cap"].str.replace(',', '').astype(float)
    df["Volume"] = df["Volume"].str.replace(',', '').astype(float)
    return df


def normalized(df):
    return (df - df.mean()) / df.std()


def load_data(k, test_number=8):
    trend_df = load_trend()
    coin_df = load_coin_mkt()
    money = coin_df['Close**'].values
    trend = normalized(trend_df).values
    coin = normalized(coin_df.drop('Close**', axis=1)).values
    trend_coin = np.concatenate((trend, coin), axis=1)

    m = trend_coin.shape[0]
    data = []
    target = []
    for i in range(m - k - 1):
        data.append(trend_coin[i:i + k, :].flatten())
        if money[i + k] >= money[i + k + 1]:  # 跌
            target.append(0)
        else:  # 漲
            target.append(1)
    data = np.array(data)
    target = np.array(target)

    train_data = data[:-test_number, :]
    train_target = target[:-test_number]

    test_data = data[-test_number:, :]
    test_target = target[-test_number:]
    test_money = money[-test_number - 1:]

    return train_data, train_target, test_data, test_target, test_money


def go(model, test_data, test_money, coin=10000):
    state = 0  # 0: 沒買, 1: 有買
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



k = 48*3
train_data, train_target, test_data, test_target, test_money = load_data(k, 48*2)

model = SVC(kernel='rbf', C = 10.0, gamma=0.1)
model.fit(train_data, train_target)

train_acc = model.score(train_data, train_target)
print ("train acc = {0}".format(train_acc))
test_acc = model.score(test_data, test_target)
print ("test acc = {0}".format(test_acc))

coin = go(model, test_data, test_money)
print ("if you throw 10000 coin, you will get {0} coin in reward.".format(coin))

