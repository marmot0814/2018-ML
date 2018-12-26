from pytrends.request import TrendReq
import json
import matplotlib.pyplot as plt
import csv


# Hyper parameter
key_words = ['bitcoin','bit coin', 'virtual currency', 'BTC']
show_plot = False
export_as_csv = True
####################

pytrend = TrendReq(hl = 'en-US', tz = 360)
pytrend.build_payload(kw_list = key_words, geo = 'TW')
data = json.loads(pytrend.interest_over_time().to_json(orient='table'))['data'][:-1]


## extract data
date = [x['date'].split('T')[0] for x in data]
value = []

for key_word in key_words:
	y = [i[key_word] for i in data]
	print(key_word)
	print(y)
	print('= = = = = = =')
	value.append(y)

## plot
if show_plot:
	for idx, y in enumerate(value):
		plt.plot(y)
		plt.title(key_words[idx])
		plt.show()

## export csv
if export_as_csv:
	with open('google_trend.csv', 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['', *date])
		for idx in range(len(key_words)):
			writer.writerow([key_words[idx], *value[idx]])
