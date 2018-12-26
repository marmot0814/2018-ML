import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import time

time.strftime("%Y%m%d", time.localtime())
res = requests.get('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end={}'.format(time.strftime("%Y%m%d", time.localtime())))
soup = BeautifulSoup(res.text, 'html.parser')
table = soup.find("table", {"class": "table"})
title = [th.text.replace('\n', '') for th in table.find('tr').find_all('th')]
table_body = table.find('tbody')
rows = table_body.find_all('tr')
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])

df = pd.DataFrame(data = data, columns = title)
df.to_csv('coin_market.csv')
