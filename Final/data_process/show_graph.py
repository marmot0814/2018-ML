import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
from sklearn import preprocessing  # normalization


def load_trend():
    csv = "../data_process/google_trend.csv"
    df = pd.read_csv(csv)
    search_sum = np.sum(df.values[:, [1, 3]], axis=1)
    #print(df.values[:, 2])
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

    print(df.head())
    #return df.drop(
    #    "Close**", axis=1).values[:, 1:], df.loc[:, "Close**"].values
    dates = df["Date"].values
    closes = df["Close**"].values

    data = {}
    data["date"] = dates
    data["close"] = closes
    data["volume"] = df["Volume"].values
    return data


if __name__ == "__main__":

    month_sum, trend_date = load_trend()
    data = load_coin_mkt()
    #print("dates.shape: ", data["date"].shape)
    #print("close.shape: ", data["close"].shape)
    for i, date in enumerate(data["date"]):
        data["date"][i] = datetime.datetime.strptime(date, '%Y-%m-%d').date()

    ## Close-date fig
    fig_close = plt.figure(1)

    # date tick
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    locator = mdates.MonthLocator()
    plt.gca().xaxis.set_major_locator(locator)
    plt.tick_params(axis='x', which='major', labelsize=5)

    plt.plot(data["date"], data["close"])
    plt.ylabel("Close")
    plt.gcf().autofmt_xdate()

    ##
    ## Volume-date fig
    fig_volume = plt.figure(2)
    # date tick
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    locator = mdates.MonthLocator()
    plt.gca().xaxis.set_major_locator(locator)
    plt.tick_params(axis='x', which='major', labelsize=5)

    plt.plot(data["date"], data["volume"])
    plt.ylabel("Volume")
    plt.gcf().autofmt_xdate()

    # 將兩者標準化放在一起看
    fig_close_volumn = plt.figure(3)
    close_norm = preprocessing.scale(data["close"])
    volume_norm = preprocessing.scale(data["volume"])

    # date tick
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    locator = mdates.MonthLocator()
    plt.gca().xaxis.set_major_locator(locator)
    plt.tick_params(axis='x', which='major', labelsize=5)

    plt.plot(data["date"], close_norm, label="normalized Close")
    plt.plot(data["date"], volume_norm, label="normalized Volume")
    plt.gcf().autofmt_xdate()
    plt.legend()

    ##
    ## trend
    fig_trend = plt.figure(4)

    # date tick
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    locator = mdates.MonthLocator()
    plt.gca().xaxis.set_major_locator(locator)
    plt.tick_params(axis='x', which='major', labelsize=5)
    month_sum_norm = preprocessing.scale(month_sum[1:])
    plt.plot(trend_date[1:], month_sum_norm, label="normalized Trends")
    plt.plot(data["date"], close_norm, label="normalized Close")
    plt.gcf().autofmt_xdate()
    plt.legend()

    plt.show()