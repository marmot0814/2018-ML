import csv, wget, random, os, math
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing, tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_score, train_test_split


# Read csv file and put feathers in a list of dict and list of class label
class Forest:

    def __init__(self, dataname):
        self.dataname = dataname
        if not os.path.exists(dataname):
            self.fetch_data()

    def fetch_data(
            self,
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    ):
        filename = self.dataname
        tmp_file = wget.download(url)
        label = "sepal length,sepal width,petal length,petal width,outcome\n"
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
            file.write(label + filtered)
        os.rename(filename, "data.csv")

    def create_data(self, train_test_ratio
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
            outcome_name = []
            for outcome in outcomes:
                outcome_name.append(outcome)
            self.outcome_name = outcome_name

        random_state = random.randint(0, 99999)
        self.data, self.test_data = train_test_split(
            data, random_state=random_state)
        self.target, self.test_target = train_test_split(
            target, random_state=random_state)

    def Kfold(self, fold):
        pass

    def create_trees(self, tree_cnt, train_test_ratio):
        self.create_data(train_test_ratio)
        # allData = open(self.dataname, 'r')
        # reader = csv.reader(allData)

        # # data headers
        # headers = next(reader)[0:4]

        # # feature
        # data = []

        # #label
        # target = []  # Y
        # outcomes = {}  # classes

        # # construct data and target
        # for row in reader:
        #     if outcomes.get(row[-1]) == None:
        #         outcomes[row[-1]] = len(outcomes)

        #     target.append(outcomes[row[-1]])

        #     col = []
        #     for i in range(0, len(row) - 1):
        #         col.append(float(row[i]))
        #     data.append(col)

        # create trees
        self.forest = []
        for tree_i in range(tree_cnt):
            # handle partial data for current tree
            tree_sample_index = random.sample(
                [x for x in range(0, len(self.data))],
                math.ceil(0.7 * len(self.data)))
            tree_data = [self.data[i] for i in tree_sample_index]
            tree_target = [self.target[i] for i in tree_sample_index]

            # shuffle the data
            shuffle_index = [x for x in range(0, len(tree_target))]
            random.shuffle(shuffle_index)
            tree_data = [x for _, x in sorted(zip(shuffle_index, tree_data))]
            tree_target = [
                x for _, x in sorted(zip(shuffle_index, tree_target))
            ]
            # generate outcome_name
            # outcome_name = []
            # for outcome in outcomes:
            #     outcome_name.append(outcome)
            # self.outcome_name = outcome_name

            # build Tree
            clf = tree.DecisionTreeClassifier(criterion='entropy')
            clf = clf.fit(tree_data, tree_target)
            self.forest.append(clf)
            # export dot file
            with open("dataOutput_{}.dot".format(tree_i), 'w') as f:
                f = tree.export_graphviz(
                    clf,
                    feature_names=self.headers,
                    class_names=self.outcome_name,
                    out_file=f)

    def predict(self, input_data):
        vote = [0, 0, 0]
        try:
            for tree in self.forest:
                vote[tree.predict(input_data)[0]] += 1
            prediction = [i for i, j in enumerate(vote) if j == max(vote)]
            if len(prediction) == 1:
                print(self.outcome_name[prediction[0]])
            else:
                print("Ties:")
                for candicate in prediction:
                    print(self.outcome_name[candicate])
        except AttributeError:
            print(
                "*** Error: Please run create_trees() before predicting! ***")


forest = Forest('data.csv')
forest.create_trees(4, 7)
forest.predict([[5.1, 3.5, 1.4, 0.2]])
# K fold validation
# scores = cross_val_score(clf, data, target, cv=5, scoring='accuracy')
# print(scores)
# print(scores.mean())