# create data from googleplaystore.csv
import csv

attributes = ['Rating', 'Reviews', 'Size', 'Installs', 'Price', 'Genres']
outcomes = {} # Category, row[1]
outcome_name = []
data = []

with open('googleplaystore.csv', 'r') as csvfile:

    rows = csv.reader(csvfile)
    next(rows)

    for row in rows:

        # extract illegal entries
        # if Rating is NaN, then drop it
        if row[2] == 'NaN':
            continue
        # only a trash data
        if row[1] == '1.9':
            continue
        # if size isn't float
        if row[4][0] == 'V':
            continue

        # construct possible outcomes
        if outcomes.get(row[1]) == None:
            outcomes[row[1]] = len(outcomes)

        # deal with genres
        genres = row[9].split(';')
        
        # deal with size
        if row[-1] == 'k':
            size = float(row[4][:-1]) * 0.001
        else:
            size = float(row[4][:-1])

        # deal with price
        if row[7] == '0':
            price = 0
        else:
            price = float(row[7][1:])

        # push into array and split different genres into different entries
        for genre in genres:
            # Rating, Reviews, Size, Installs, Price, Genres
            entry = [row[2], int(row[3]), size, int(row[5][:-1].replace(',','')), price, genre]
            data.append(entry)

    for outcome in outcomes:
            outcome_name.append(outcome)

print(data)
