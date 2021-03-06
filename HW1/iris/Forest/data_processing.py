import csv, wget, random, os, math
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing, tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read csv file and put feath3ers in a list of dict and list of class label
class Forest:

    def __init__(
            self,
            dataname,
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            label=""):
        self.dataname = dataname
        self.label = label
        if not os.path.exists(dataname):
            self.fetch_data(url)

    def fetch_data(self, url):
        filename = self.dataname
        tmp_file = wget.download(url)

        # read data to tmp
        with open(tmp_file, 'r') as file:
            tmp = file.read()
        # remove file inorder to recreate with disired filename
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        filtered = ''
        # remove blank line
        for line in tmp.split('\n'):
            if line.strip():
                filtered += line + '\n'
        with open(filename, 'w+') as file:
            file.write(self.label + filtered)
        os.rename(filename, "data.csv")

    def create_data(self, train_test_ratio=7
                   ):  # if train_test_ratio = 7, then train:test = 7:3
        allData = open(self.dataname, 'r')
        reader = csv.reader(allData)

        # data headers
        self.headers = next(reader)[0:-1]

        # feature
        data = []

        #label
        target = []  # Y
        outcomes = {}  # classes

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
            outcome_name = [0 for x in range(len(outcomes))]
            for outcome in outcomes:
                outcome_name[outcomes[outcome]] = outcome
            self.outcome_name = outcome_name

        # shuffle data
        data, target = shuffle(data, target)

        # full_data for Validation
        self.full_data = data
        self.full_target = target
        random_state = random.randint(0, 99999)
        data, test_data = train_test_split(data, random_state=random_state)
        target, test_target = train_test_split(
            target, random_state=random_state)

        return data, target, test_data, test_target

    def KFold(self, fold=10, tree_cnt=1):
        # resubstitution validation
        if (fold == 1):
            clf = self.create_trees(self.full_data, self.full_target, tree_cnt)
            predictions = self.predict(clf, self.full_data)
            score = sum([
                predictions[i] == self.outcome_name[self.full_target[i]]
                for i in range(len(predictions))
            ]) / len(predictions)

            pred = predictions
            true = [self.outcome_name[self.full_target[i]] for i in range(len(predictions))]

            C = confusion_matrix(true, pred, labels=self.outcome_name)
            report = classification_report(true, pred, target_names = self.outcome_name)
            print (report)
            df_cm = pd.DataFrame(C, index = self.outcome_name, columns = self.outcome_name)
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=True,  fmt='g', cmap='Blues')
            plt.show() 

            return score, score / fold

        self.create_data()
        kf = KFold(n_splits=fold, shuffle=False)
        scores = []
    
        pred = []
        true = []

        for train, test in kf.split(self.full_data):
            # use the fold to create_trees
            clfs = self.create_trees([self.full_data[i] for i in train],
                                     [self.full_target[i] for i in train],
                                     tree_cnt)
            # test clfs
            test_data = [self.full_data[i] for i in test]
            test_target = [self.full_target[i] for i in test]
            predictions = self.predict(clfs, test_data)

            pred += predictions
            true += [self.outcome_name[test_target[i]] for i in range(len(test_target))]

            scores.append(
                sum([
                    predictions[i] == self.outcome_name[test_target[i]]
                    for i in range(len(predictions))
                ]) / len(predictions))
        
        report = classification_report(true, pred, target_names = self.outcome_name)
        print (report)

        C = confusion_matrix(true, pred, labels=self.outcome_name)
        df_cm = pd.DataFrame(C, index = self.outcome_name, columns = self.outcome_name)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True,  fmt='g', cmap='Blues')
        plt.show() 
        return (scores, sum(scores) / fold)

    def create_trees(self, data, target, tree_cnt=1):
        forest = []
        self.picked_column = []
        # handle partial feature for current tree
        for tree_i in range(tree_cnt):
            #tree_sample_index = random.sample([x for x in range(0, len(data))],
            #                                  math.ceil(0.7 * len(data)))
            if (tree_cnt != 1):
                picked_column = random.sample(
                    [x for x in range(0, len(self.headers))],
                    random.randint(1, len(self.headers)))
                picked_column.sort()
                self.picked_column.append(picked_column)
            else:
                # pick all column
                picked_column = [i for i, _ in enumerate(self.headers)]
                self.picked_column.append(picked_column)
            headers = [self.headers[i] for i in picked_column]
            tree_data = []
            for i in range(len(data)):
                data_row = [data[i][j] for j in picked_column]
                tree_data.append(data_row)
            tree_target = target

            # build Tree
            clf = tree.DecisionTreeClassifier(criterion='entropy')
            clf = clf.fit(tree_data, tree_target)
            forest.append(clf)
            # export dot file
            with open("dataOutput_{}.dot".format(tree_i), 'w') as f:
                f = tree.export_graphviz(
                    clf,
                    feature_names=headers,
                    class_names=self.outcome_name,
                    out_file=f)
        return forest

    def predict(self, forest, input_data):
        predictions = []
        try:
            for data_i in range(len(input_data)):
                vote = [0 for x in range(len(self.outcome_name))]
                for i, tree in enumerate(forest):
                    # handle partial feature in the forest
                    partial_featured_data = [
                        input_data[data_i][j] for j in self.picked_column[i]
                    ]
                    vote[tree.predict([partial_featured_data])[0]] += 1
                    #print(vote)
                prediction = [i for i, j in enumerate(vote) if j == max(vote)]
                predictions.append(self.outcome_name[prediction[0]])
            return predictions
        except AttributeError:
            print(
                "*** Error: Please run create_trees() before predicting! ***")
