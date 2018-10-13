import csv
import pandas as pd

data = pd.read_csv('googleplaystore_user_reviews.csv')

# Parse App Sentiment Sentiment_Polarity Sentiment_Subjectivity
data = data.iloc[:,[0,2,3,4]]

# drop nan if any
data.dropna(axis=0,how='any')

# drop null if any
null_data = data.isnull()
result = null_data.iloc[:,0]
for i in range(1,4):
    result = result | null_data.iloc[:,i]
result = ~result
data = data.loc[result,:]

# to csv
data.to_csv('new_googleplaystore_user_reviews.csv',index=False)