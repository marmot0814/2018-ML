import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import math
from sklearn.svm import SVC
from sklearn.utils import shuffle


# load_trend() return the sum of search in every month
# starting at 2014-Jan to 2018-Dec
def load_trend():
    csv = "../data_process/google_trend.csv"
    df = pd.read_csv(csv)
    search_sum = np.sum(df.values[:, 1:], axis=0)
    prev_month = 1
    s = 0
    month_sum = []
    for i, day in enumerate(df.columns.values[2:]):
        month = datetime.datetime.strptime(day, "%Y-%m-%d").month

        if (prev_month != month or i == len(df.columns.values[2:]) - 1):
            month_sum.append(s)
            prev_month = month
            s = 0

        s += search_sum[i]

    return np.array(month_sum)


def load_coin_mkt():
    csv = "../data_process/coin_market.csv"
    df = pd.read_csv(csv)#.drop("Volume", axis=1)
    df["Date"] = df["Date"].apply(
        lambda date: datetime.datetime.strptime(date, '%b %d, %Y').strftime("%Y-%m-%d")
    )
    df = df.sort_values(by="Date")
    df = df.loc[df['Date'] > "2014-00-00"]
    df["Market Cap"] = df["Market Cap"].str.replace(',', '').astype(float)
    df["Volume"] = df["Volume"].str.replace(',', '').astype(float)
    #print(df.head())
    #print(df['Date'] > "2014-00-00")

    #datas = []
    #print(df.values[:, 1:])
    #print(df.columns)
    print(df.head())
    return df.drop(
        "Close**", axis=1).values[:, 2:], df.loc[:, "Close**"].values


if __name__ == "__main__":
    #print(len(load_trend()))
    coin_datas, coin_target = load_coin_mkt()
    price = np.array(coin_target[10:])
    #print(coin_target)
    #print(coin_datas)
    prev_ten_data = []
    prev_ten_target = []

    tmp_target = [0]
    for i in range(len(coin_target) - 1):
        tmp_target.append(int(coin_target[i + 1] - coin_target[i] > 0))
    coin_target = tmp_target

    # 每十天的資料加總 除以10
    for index in range(len(coin_datas) - 11 + 1):
        i = index
        #ten_day_data = np.sum(coin_datas[i:i + 10], axis=0) / 10
        ten_day_data = np.array(coin_datas[i:i + 10]).flatten()
        ten_day_target = coin_target[i + 10]

        prev_ten_data.append(ten_day_data)
        prev_ten_target.append(ten_day_target)

    # shuffle
    randomize = np.arange(len(prev_ten_data))
    np.random.shuffle(randomize)
    prev_ten_data = np.array(prev_ten_data)[randomize]
    prev_ten_target = np.array(prev_ten_target)[randomize]
    price = price[randomize]

    # 切成 train & test data
    train_data = prev_ten_data[0:int(len(prev_ten_data) / 3 * 2)]
    train_target = prev_ten_target[0:int(len(prev_ten_target) / 3 * 2)]
    print(train_target)

    test_data = prev_ten_data[int(len(prev_ten_data) / 3 * 2):-1]
    test_target = prev_ten_target[int(len(prev_ten_target) / 3 * 2):-1]
    price = price[int(len(prev_ten_target) / 3 * 2):-1]
    #reg = LinearRegression().fit(train_data, train_target)
    reg = SVC(gamma='auto').fit(train_data, train_target)
    #print("weight: ", reg.coef_)
    #print("train score: ", reg.score(train_data, train_target))
    #print("test score: ", reg.score(test_data, test_target))
    #print(test_data[-1])
    #print("test_target", test_target[-1])
    #print("tomorrow: ", reg.predict([np.array(coin_datas[-12:-2]).flatten()]))
    #error = 0
    dollar = 10000
    coin = 0
    for i, _ in enumerate(test_data):
        predict = reg.predict([test_data[i]])
        print(predict)
        if (i == len(test_data) - 1):
            dollar += coin * price[i]
            break
        if predict:
            coin += int(dollar / price[i])
            dollar = dollar % price[i]
        else:
            dollar += coin * price[i]
            coin = 0

    #print("coin left: ", coin)
    print("dollar left: ", dollar)
    #print(
    #    np.sum(np.square(reg.predict(test_data) - test_target)) /
    #    len(test_data))
    #print(np.sum(test_target - reg.predict(test_data)) / len(test_data))
