from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# Read csv file and put feathers in a list of dict and list of class label
allData = open('data.csv', 'r')
reader = csv.reader(allData)

# data headers
headers = next(reader)

# feature
dummyX = []

#label
labelList = []

for row in reader:
    labelList.append(row[-1])
    col = []
    for i in range(0, len(row) - 1):
        col.append(float(row[i]))
    dummyX.append(col)

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

# print (dummyX)
# print (dummyY)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)

with open("dataOutput.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names = headers[0:4], out_file = f)
