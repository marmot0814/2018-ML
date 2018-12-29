import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import time
import datetime

time.strftime("%Y%m%d", time.localtime())
res = requests.get('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20131229&end=20181216')
soup = BeautifulSoup(res.text, 'html.parser')
table = soup.find("table", {"class": "table"})
title = [th.text.replace('\n', '') for th in table.find('tr').find_all('th')]
table_body = table.find('tbody')
rows = table_body.find_all('tr')
data = []
cnt = 6
for row in rows:
    cnt += 1
    cnt %= 7
    if cnt != 0:
        continue
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])

df = pd.DataFrame(data = data, columns = title)
df.to_csv('coin_market.csv')
