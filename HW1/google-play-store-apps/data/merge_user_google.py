import pandas as pd

data = pd.read_csv('new_googleplaystore_user_reviews.csv')
data2 = pd.read_csv('googleplaystore.csv')

polarity = data.groupby('App')['Sentiment_Polarity'].mean().to_frame()
count = data.groupby('App')['Sentiment_Polarity'].count().to_frame()

polarity = polarity.rename(columns={'Sentiment_Polarity':'comment_mean'})
count = count.rename(columns={'Sentiment_Polarity':'comment_count'})

res = pd.merge(data2, polarity, on = 'App', how='outer')
res = pd.merge(res, count, on ='App', how='outer')

null_data = res.isnull()
res.loc[null_data.iloc[:,-2],'comment_mean']=0
res.loc[null_data.iloc[:,-1],'comment_count']=0

res.to_csv('merged_googleplaystore.csv',index=False)