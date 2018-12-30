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
    df = pd.read_csv(csv)[['BTC', 'bitcoin']]
    return (df - df.mean()) / df.std()


def load_coin_mkt():
    csv = "../data_process/coin_market.csv"
    df = pd.read_csv(csv).drop("Date", axis = 1).drop('Unnamed: 0', axis = 1)
    df["Market Cap"] = df["Market Cap"].str.replace(',', '').astype(float)
    df["Volume"] = df["Volume"].str.replace(',', '').astype(float)
    return (df - df.mean()) / df.std()

def load_data(k = 8, ratio = 0.7):
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
        target.append(data_[i+k, TARGET_INDEX])
    data = np.array(data)
    target = np.array(target)

    x = round(m * ratio)
    train_data = data[:x, :]
    train_target = data[:x, :]
    test_data = data[x:, :]
    test_target = data[:x, :]
    return train_data, train_target, test_data, test_target

train_data, train_target, test_data, test_target = load_data()
