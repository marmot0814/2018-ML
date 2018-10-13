import pandas as pd

data = pd.read_csv('new_googleplaystore_user_reviews.csv')
data2 = pd.read_csv('googleplaystore.csv')

polarity = data.groupby('App')['Sentiment_Polarity'].mean().to_frame()
count = data.groupby('App')['Sentiment_Polarity'].count().to_frame()

polarity = polarity.rename(columns={'Sentiment_Polarity':'comment_mean'})
count = count.rename(columns={'Sentiment_Polarity':'comment_count'})

res = pd.merge(data2, polarity, on = 'App')
res = pd.merge(res, count, on ='App')

res.to_csv('merged_googleplaystore.csv',index=False)