import wget
import os
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
if os.path.exists("data.csv"):
    os.remove("data.csv")
filename = wget.download(url)
label = "sepal length,sepal width,petal length,petal width,outcome\n"
with open(filename, 'r') as file:
    tmp = file.read()

filtered = ''
# remove blank line
for line in tmp.split('\n'):
    if line.strip():
        filtered += line + '\n'
with open(filename, 'w+') as file:
    file.write(label + filtered)
os.rename(filename, "data.csv")
