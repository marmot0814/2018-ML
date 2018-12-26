import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression


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
    df = pd.read_csv(csv).drop("Volume", axis=1)
    df["Date"] = df["Date"].apply(
        lambda date: datetime.datetime.strptime(date, '%b %d, %Y').strftime("%Y-%m-%d")
    )
    df = df.sort_values(by="Date")
    df["Market Cap"] = df["Market Cap"].str.replace(',', '').astype(float)
    #print(df.head())
    #print(df['Date'] > "2014-00-00")
    #df = df.loc[df['Date'] > "2014-00-00"]
    #datas = []
    #print(df.values[:, 1:])
    #print(df.columns)
    return df.drop(
        "Close**", axis=1).values[:, 2:], df.loc[:, "Close**"].values


if __name__ == "__main__":
    #print(len(load_trend()))
    coin_datas, coin_target = load_coin_mkt()
    #print(coin_target)
    #print(coin_datas)
    prev_ten_data = []
    prev_ten_target = []
    for index in range(len(coin_datas) - 10 + 1):
        i = index + 10
        ten_day_data = np.sum(coin_datas[i:i + 9], axis=0) / 10
        ten_day_target = np.sum(coin_target[i:i + 9], axis=0) / 10
        prev_ten_data.append(ten_day_data)
        prev_ten_target.append(ten_day_target)

    train_data = prev_ten_data[1:int(len(prev_ten_data) / 3 * 2)]
    train_target = prev_ten_target[1:int(len(prev_ten_target) / 3 * 2)]

    test_data = prev_ten_data[int(len(prev_ten_data) / 3 * 2):]
    test_target = prev_ten_target[int(len(prev_ten_target) / 3 * 2):]
    reg = LinearRegression().fit(train_data, train_target)
    print("train score: ", reg.score(train_data, train_target))
    print("test score: ", reg.score(test_data, test_target))
