import sys
sys.path.append("../iris/Forest")
from data_processing import Forest

forest = Forest('data.csv')
data, target, test_data, test_target = forest.create_data()

scores, avg_score = forest.KFold(5)
print (scores)
print (avg_score)

clfs = forest.create_trees(data, target, tree_cnt=5)
