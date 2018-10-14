import sys
sys.path.append("../iris/Forest")
from data_processing import Forest

forest = Forest('data/new_googleplaystore.csv')
data, target, test_data, test_target = forest.create_data()

acc = [[0 for x in range(20)] for x in range(20)]
for K in range(20):
    for T in range(20):
        print (K + 1)
        print (T + 1)
        scores, avg_score = forest.KFold(K + 1, tree_cnt = T + 1)
        acc[K][T] = avg_score

print (acc)
