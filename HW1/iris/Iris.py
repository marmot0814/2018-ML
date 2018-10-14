from Forest.data_processing import Forest

forest = Forest(
    'data.csv',
    label="sepal length,sepal width,petal length,petal width,outcome\n")
data, target, test_data, test_target = forest.create_data()
scores, avg_score = forest.KFold(1, tree_cnt=1)
print(scores)
print(avg_score)
clfs = forest.create_trees(data, target, tree_cnt=1)
print(forest.predict(clfs, [[5.1, 3.5, 1.4, 0.2]]))
