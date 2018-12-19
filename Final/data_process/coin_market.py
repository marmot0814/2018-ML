import requests
import pandas
import matplotlib.pyplot as plt

res = requests.get('https://graphs2.coinmarketcap.com/currencies/bitcoin/')
d = res.json()
df = pandas.DataFrame(d)
print (df)
