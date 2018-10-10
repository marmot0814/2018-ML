from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_score
import random
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

# shuffle the data
shuffle_index = [x for x in range(0, len(target))]
random.shuffle(shuffle_index)
data = [x for _, x in sorted(zip(shuffle_index, data))]
target = [x for _, x in sorted(zip(shuffle_index, target))]

# generate outcome_name
outcome_name = []
for outcome in outcomes:
    outcome_name.append(outcome)

# build Tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(data, target)
# export dot file
with open("dataOutput.dot", 'w') as f:
    f = tree.export_graphviz(
        clf, feature_names=headers, class_names=outcome_name, out_file=f)

# K fold validation
scores = cross_val_score(clf, data, target, cv=5, scoring='accuracy')
print(scores)
print(scores.mean())