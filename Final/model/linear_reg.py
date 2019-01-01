import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import math
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


# date starts from 2013 Dec,
def load_trend():
    csv = "../data_process/google_trend.csv"
    df = pd.read_csv(csv)
    search_sum = np.sum(df.values[:, [1, 3]], axis=1)
    prev_month = 1
    s = 0
    month_sum = []
    date = []
    for i, day in enumerate(df.values[:, 2]):

        month = datetime.datetime.strptime(day, "%Y-%m-%d").month

        if (prev_month != month or i == len(df.columns.values[2:]) - 1):
            month_sum.append(s)
            prev_month = month
            date.append(datetime.datetime.strptime(day, "%Y-%m-%d").date())
            s = 0

        s += search_sum[i]

    return np.flip(np.array(month_sum)), np.flip(np.array(date))


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

    # print(df.head())
    return df.drop(
        "Close**", axis=1).values[:, 1:], df.loc[:, "Close**"].values


if __name__ == "__main__":
    data, close = load_coin_mkt()
    date, data = data[0:-2, 0], data[0:-2, 1:]  # date start from 2014 Jan
    month_sum, trend_date = load_trend()  # date start from 2013 Dec

    # coin_mkt, trend日期對齊 都從2014 Jan開始 => 1:
    # date是每 8 筆資料代表兩個月 不過最後一個月只有3筆
    # 每兩個月的資料預測下個月的第一筆 所以我們讓最後一個月的資料只留下第一筆當作target => :-2
    # trend_date 一筆資料一個月

    month_sum = month_sum[1:-1]
    trend_date = trend_date[1:-1]

    ## 兩個月前的data 配target

    two_month_data, two_month_target = [], []
    print(len(date))
    print(len(month_sum))
    for i in range(8, len(date)):
        d = []
        d.extend(np.concatenate(data[i - 8:i], axis=None).tolist())
        #d.extend(month_sum[int((i - 4) / 4):int(i / 4)].tolist())
        two_month_data.append(d)
        #print(len(d))
        two_month_target.append(close[i])

    # K-fold
    kf = KFold(n_splits=8)
    two_month_data = np.array(two_month_data)
    print(two_month_data.shape)
    two_month_target = np.array(two_month_target)
    for train_index, test_index in kf.split(two_month_data):
        train_data, train_target = two_month_data[
            train_index], two_month_target[train_index]
        test_data, test_target = two_month_data[test_index], two_month_target[
            test_index]
        reg = LinearRegression().fit(train_data, train_target)
        #print("train score: ", reg.score(train_data, train_target))
        print("test score: ", reg.score(test_data, test_target))
    """
    coin_datas, coin_target = load_coin_mkt()
    price = np.array(coin_target[10:])
    prev_ten_data = []
    prev_ten_target = []

    # tmp_target = [0]
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

    test_data = prev_ten_data[int(len(prev_ten_data) / 3 * 2):]
    test_target = prev_ten_target[int(len(prev_ten_target) / 3 * 2):]
    price = price[int(len(prev_ten_target) / 3 * 2):]
    reg = LinearRegression().fit(train_data, train_target)
    #reg = SVC(gamma='auto').fit(train_data, train_target)
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
    """