import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from collections import Counter
import pylab as pl
from operator import itemgetter

data = pd.read_csv('./merged_googleplaystore.csv')
data2 = pd.read_csv('./new_googleplaystore_user_reviews.csv')

polarity_avg = data2.groupby('App')['Sentiment_Polarity'].mean().to_frame()\
	.rename(columns={'Sentiment_Polarity':'comment_mean'}).fillna(0)\
	.sort_values('comment_mean')
count = data2.groupby('App')['Sentiment_Polarity'].count().to_frame()\
	.rename(columns={'Sentiment_Polarity':'comment_count'})\
	.sort_values('comment_count')


app_cnt = data.groupby('App')['App'].count().shape[0]

category_cnt = data.groupby('Category')['Category'].count().shape[0]

rating_avg = data.loc[:,'Rating'].mean()
rating_std = data.loc[:,'Rating'].std()

review_avg = data.loc[:,'Reviews'].mean()
review_std = data.loc[:,'Reviews'].std()

app_size_clean = data.loc[:,'Size'].where(data.loc[:,'Size'] != 'Varies with device'\
	,0).str.replace('M','000000').str.replace('k','000').fillna(0)
app_size_clean = pd.to_numeric(app_size_clean)
app_size_clean = app_size_clean.to_frame()

app_price_clean = data.loc[:,'Price'].str.replace('$','')
app_price_clean = pd.to_numeric(app_price_clean).to_frame().dropna()

## Final
### App Count
# print(f'Total APP counts: {app_cnt}')

### Category
# print(f'Total category counts: {category_cnt}')

### size
# print(app_size_clean.loc[:,'Size'].describe())

### Rating
# print(data.loc[:,'Rating'].describe())
# temp = pd.to_numeric(data.loc[:,'Rating']).fillna(0).sort_values()
# print(temp)
# plt.hist(temp)
# plt.title('App Rating')
# plt.show()


### Review
# print(data.loc[:,'Reviews'].describe())


### Installs
# labels, values = zip(*sorted(Counter(data.loc[:,'Installs'].fillna('nan')).items(), key = itemgetter(1)))
# plt.bar(labels, values)
# plt.title('App Installs')
# pl.xticks(rotation=20)
# plt.show()


### Price
# print("Total free app: ",app_price_clean.loc[app_price_clean.loc[:,'Price']==0,:].count()[0])
# temp = app_price_clean.loc[app_price_clean.loc[:,'Price']!=0,:].sort_values('Price')
# print(temp)


### Content Rating
# labels, values = zip(*sorted(Counter(data.loc[:,'Content Rating'].fillna('nan')).items()))
# temp = pd.DataFrame({
# 		"labels": labels,
# 		"values": values
# 	})
# print(temp.sort_values('values',ascending=False))
# plt.bar(labels, values)
# plt.title('App content rating')
# pl.xticks(rotation=20)
# plt.show()


### Genres
# temp = []
# for i in list(data.loc[:,'Genres']):
# 	if type(i)!=float or ~np.isnan(i):
# 		for j in i.split(';'):
# 			temp.append(j)
# labels, values = zip(*sorted(Counter(temp).items()))
# temp = pd.DataFrame({
# 		"labels": labels,
# 		"values": values
# 	})
# print(temp.sort_values('values',ascending=False))
# print('Total labels count: ',len(temp))


### app user review
# comment_count = data2.groupby('App')['App'].count().to_frame()
# print(comment_count.describe())

### app Sentiment_Polarity
temp = data2.groupby('App')['Sentiment_Polarity'].mean()
print(temp)
print(temp.describe())