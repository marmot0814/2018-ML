import pandas as pd
import numpy as np
df = pd.read_csv('data_noah.csv')
# print(df.columns)
# print(df.count(axis='rows'))
df = df[['id', 'x', 'y', 'pitch_type']]
#df = df[['x', 'y', 'pitch_type']]
# print(len(df.index))

# print(df)
global kmeansNumppy, startingPoint, oldStartingPoint
kmeansNumppy = df
startingPoint = df

# kmeans = np.zeros((len(df.index), 2))


def Distance(x1, x2, y1, y2):
    return (x1 - x2)**2 + (y1 - y2)**2


def Calculation(dfStartingPoint):
    global kmeansNumppy
    for index, row in kmeansNumppy.iterrows():
        distanceList = [
            Distance(row['x'], dfStartingPoint['x'][i], row['y'],
                     dfStartingPoint['y'][i]) for i in range(3)
        ]
        kmeansNumppy.loc[[index], ['pitch_type']] = distanceList.index(
            min(distanceList))
    # print(kmeansNumppy)
    return kmeansNumppy


def FindNewStartingPoint():
    global startingPoint
    startingPoint = kmeansNumppy[['x', 'y',
                                  'pitch_type']].groupby('pitch_type').sum()
    sumAllCount = kmeansNumppy.groupby('pitch_type').count()
    # print(sumAllCount)
    sumAllCount = sumAllCount.values
    for index, row in startingPoint.iterrows():
        startingPoint.loc[[index], ['x']] = startingPoint.loc[
            [index], ['x']] / sumAllCount.item((index, 1))
        startingPoint.loc[[index], ['y']] = startingPoint.loc[
            [index], ['y']] / sumAllCount.item((index, 1))
    print(startingPoint)


while True:
    Calculation(startingPoint)
    oldStartingPoint = startingPoint.values
    FindNewStartingPoint()
    if (np.array_equal(oldStartingPoint, startingPoint.values)):
        break
print('done')
