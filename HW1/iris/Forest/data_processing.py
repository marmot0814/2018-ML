import csv, wget, random, os, math
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing, tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_score, train_test_split, KFold


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
        self.headers = next(reader)[0:4]

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
        self.full_data = data
        self.full_target = target
        random_state = random.randint(0, 99999)
        data, test_data = train_test_split(data, random_state=random_state)
        target, test_target = train_test_split(
            target, random_state=random_state)

        return data, target, test_data, test_target

    def KFold(self, fold=10, tree_cnt=1):
        self.create_data()
        kf = KFold(n_splits=fold, shuffle=False)
        scores = []
        for train, test in kf.split(self.full_data):
            # use the fold to create_trees
            clfs = self.create_trees([self.full_data[i] for i in train],
                                     [self.full_target[i] for i in train],
                                     tree_cnt)
            # test clfs
            test_data = [self.full_data[i] for i in test]
            test_target = [self.full_target[i] for i in test]
            predictions = self.predict(clfs, test_data)
            scores.append(
                sum([
                    predictions[i] == self.outcome_name[test_target[i]]
                    for i in range(len(predictions))
                ]) / len(predictions))

        return (scores, sum(scores) / fold)

    def create_trees(self, data, target, tree_cnt=1):
        forest = []
        for tree_i in range(tree_cnt):
            # handle partial data for current tree
            tree_sample_index = random.sample([x for x in range(0, len(data))],
                                              math.ceil(0.7 * len(data)))
            tree_data = [data[i] for i in tree_sample_index]
            tree_target = [target[i] for i in tree_sample_index]

            # # shuffle the data
            # shuffle_index = [x for x in range(0, len(tree_target))]
            # random.shuffle(shuffle_index)
            # tree_data = [x for _, x in sorted(zip(shuffle_index, tree_data))]
            # tree_target = [
            #     x for _, x in sorted(zip(shuffle_index, tree_target))
            # ]

            # build Tree
            clf = tree.DecisionTreeClassifier(criterion='entropy')
            clf = clf.fit(tree_data, tree_target)
            forest.append(clf)
            # export dot file
            with open("dataOutput_{}.dot".format(tree_i), 'w') as f:
                f = tree.export_graphviz(
                    clf,
                    feature_names=self.headers,
                    class_names=self.outcome_name,
                    out_file=f)
        return forest

    def predict(self, forest, input_data):
        predictions = []
        try:
            for data_i in range(len(input_data)):
                vote = [0, 0, 0]
                for tree in forest:
                    vote[tree.predict([input_data[data_i]])[0]] += 1
                prediction = [i for i, j in enumerate(vote) if j == max(vote)]
                predictions.append(self.outcome_name[prediction[0]])
            return predictions
        except AttributeError:
            print(
                "*** Error: Please run create_trees() before predicting! ***")
