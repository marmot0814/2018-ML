from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# Read csv file and put feathers in a list of dict and list of class label
allData = open('data.csv', 'r')
reader = csv.reader(allData)

# data headers
headers = next(reader)[0:4]

# feature
data = []

#label
target = []
outcomes = {}

# construct data and target
for row in reader:
    if outcomes.get(row[-1]) == None:
        outcomes[row[-1]] = len(outcomes)

    target.append(outcomes[row[-1]])

    col = []
    for i in range(0, len(row) - 1):
        col.append(float(row[i]))
    data.append(col)

# generate outcome_name
outcome_name = []
for outcome in outcomes:
    outcome_name.append(outcome)

# build Tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(data, target)

# export dot file
with open("dataOutput.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names = headers, class_names = outcome_name, out_file = f)
