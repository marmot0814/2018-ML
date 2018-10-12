import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
import numpy as np

csv_file = open('googleplaystore.csv')
reader = csv.reader(csv_file)
headers = next(reader)
headers_dict = {}
for index in range(len(headers)):
    headers_dict[headers[index]] = index

target = []
target_outcomes = {}

data_dict = []

for row in reader:
    
    # A component in the data_dict list
    data_dict_row = {}    
    
    # Clear the trash
    if row[headers_dict['Category']] == '1.9':
        continue
   
    # Clear NaN
    if row[headers_dict['Rating']] == 'NaN':
        continue

    # Clear Varies with device
    if row[headers_dict['Size']][0] == 'V':
        continue

    # Rating
    data_dict_row['Rating'] = float(row[headers_dict['Rating']])

    # Reviews
    data_dict_row['Reviews'] = int(row[headers_dict['Reviews']])

    # Size

    data_dict_row['Size'] = float(row[headers_dict['Size']][:-1])
    if row[headers_dict['Size']][-1] == 'M':
        data_dict_row['Size'] *= 1000

    # Installs
    data_dict_row['Installs'] = int(row[headers_dict['Installs']][:-1].replace(',', ''))

    # Price
    if row[headers_dict['Price']][0] == '$':
        row[headers_dict['Price']] = row[headers_dict['Price']][1:]
    data_dict_row['Price'] = float(row[headers_dict['Price']])

    # Genres
    Genres = row[headers_dict['Genres']].split(';')
    for genres in Genres:
        data_dict_row[genres] = True

    # target
    if target_outcomes.get(row[1]) == None:
        target_outcomes[row[1]] = len(target_outcomes)
    target.append(target_outcomes[row[1]])

    # data
    data_dict.append(data_dict_row)

# Data
vec = DictVectorizer()
data = vec.fit_transform(data_dict).toarray()

# Data Name
data_name = vec.get_feature_names()

# Target
target = np.asarray(target)

#Target Name
target_name = [0 for x in range(len(target_outcomes))]
for target_outcome in target_outcomes:
    target_name[target_outcomes[target_outcome]] = target_outcome
    

# print ("Data Name")
# print (data_name)
# print ("Data")
# print (data)
# print ("Target Name")
# print (target_name)
# print ("target")
# print (target)

# output csv file
f = open("new_googleplaystore.csv", "w")
line = ""
for feature in data_name:
    line += feature
    line += ','
line += "outcome"
line += '\n'
f.write(line)
for index in range(len(data)):
    row = data[index]
    line = ""
    for item in row:
        line += str(item)
        line += ','
    line += str(target_name[target[index]])
    line += '\n';
    f.write(line)

