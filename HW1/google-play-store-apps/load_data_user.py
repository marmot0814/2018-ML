import csv

attributes = ['App', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
outcomes = ['Positive', 'Negative', 'Neutral']
data = []

with open('googleplaystore_user_reviews.csv', 'r') as csvfile:

    rows = csv.reader(csvfile)
    next(rows)

    for row in rows:
        
        # extract illegal entries
        if row[0] != row[0] or row[3] != row[3] or row[4] != row[4] :
            continue
        if row[2] != 'Positive' and row[2] != 'Negative' and row[2] != 'Neutral' :
            continue

        # keep 3 attributes
        row = [row[0], float(row[3]), float(row[4])]

        # push into array
        data.append(row)

print(data)
