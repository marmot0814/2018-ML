import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import csv

MPa = 'Concrete compressive strength(MPa, megapascals) '

def load_file(filename):
	with open(filename, newline='') as csvfile:
		reader = list(csv.DictReader(csvfile))
		datas = [list(i.values()) for i in reader]
		keys = list(reader[0].keys())
		return datas, keys, keys.index(MPa)
def gradient_descent(X, Y, learning_rate, epoch):
	a, b = 0.0, 0.0
	num = len(Y)
	for i in range(epoch):
		y = a*X + b
		cost = sum([i**2 for i in (Y-y)])/num
		da = -(2/num)*sum(X*(Y-y))
		db = -(2/num)*sum(Y-y)
		# softmax # overflow
		da = max(da, -1) if da < 0 else min(da,1)
		db = max(db, -1) if db < 0 else min(db,1)
		a -= learning_rate*da
		b -= learning_rate*db
		# print(da, db, a, b, cost)
	return a, b

def p1(datas, keys, ind):
	for i in range(len(keys)):
		data = np.array(datas)[:,i].astype(np.float)
		pidt = np.array(datas)[:,ind].astype(np.float)

		lm = LinearRegression()
		lm.fit(np.reshape(data,(len(data),1)),np.reshape(pidt,(len(pidt),1)))
		print(lm.coef_,lm.intercept_,keys[i]) # 印出係數 截距
		'''
		# 新的氣溫
		to_be_predicted = np.array([30])
		predicted_sales = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))
		# 預測的冰紅茶銷量
		print(predicted_sales)
		# 視覺化
		plt.scatter(data, pidt, color='black')
		plt.plot(data, lm.predict(np.reshape(data, (len(data), 1))), color='blue', linewidth=3)
		# plt.plot(to_be_predicted, predicted_sales, color = 'red', marker = '^', markersize = 10)
		ax = plt.gca()
		ax.set_aspect(1)
		plt.show()
		'''
def p2(datas, keys, ind, epoch):
	for i in range(len(keys)):
	# for i in range(1):
		data = np.array(datas)[:,i].astype(np.float)
		pidt = np.array(datas)[:,ind].astype(np.float)
		a, b = gradient_descent(data, pidt, 0.001, epoch)
		print(a, b, keys[i])

def main():
	datas, keys, ind = load_file('Concrete_Data.csv')
	# for i in range(len(keys)):
	print('Problem 1:')
	p1(datas, keys, ind)
	print('=============================')
	epoch = 100000
	print('Problem 2:')
	p2(datas, keys, ind, epoch)

if __name__ == "__main__":
    main()
