from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import charts

def target_count(iris,save):
	tmp = [np.sum(iris['target']==i) for i in range(3)]
	filename = 'target' if save is 1 else ""
	charts.pie(iris['target_names'],tmp,title='target',filename=filename)
def feature_count(i,x,feature,save):
	tmp_float = [j[i] for j in x]
	# print(np.mean(tmp_float),np.var(tmp_float))
	y = np.unique(tmp_float)
	z = [np.sum(tmp_float==j) for j in y]
	filename = feature+".png" if save is 1 else ""
	charts.bar(y,z,np.mean(tmp_float),np.var(tmp_float),title=feature,filename=filename)
def main():
	iris = load_iris()
	target_count(iris,1)
	for i in range(4):
		feature_count(i,iris.data,iris['feature_names'][i],1)

if __name__ == "__main__":
    main()
