from Forest.data_processing import Forest

forest = Forest(
    'data.csv',
    label="sepal length,sepal width,petal length,petal width,outcome\n")
data, target, test_data, test_target = forest.create_data()


K = 2
scores, avg_score = forest.KFold(K, tree_cnt=1)
print(f'{K}-folds score: {scores}')
print(f'average score: {avg_score}')
clfs = forest.create_trees(data, target, tree_cnt=1)
print(forest.predict(clfs, [[5.1, 3.5, 1.4, 0.2]]))