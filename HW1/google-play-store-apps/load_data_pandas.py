import pandas as pd

df = pd.read_csv('googleplaystore.csv')[['Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Price', 'Genres']]

for row in df.iterrows():
    index, row = row[0], row[1]

    # if Rating is NaN, then drop it
    if row['Rating'] != row['Rating']:
        df = df.drop(index)
        continue

    # only a trash data
    if row['Category'] == '1.9':
        df = df.drop(index)
        continue

    # convert str to int
    df.loc[index, 'Installs'] = int(row['Installs'][:len(row['Installs']) - 1].replace(',',''))

    # if Size isn't flaot, then drop it
    if row['Size'][0] == 'V':
        df = df.drop(index)
        continue
    df.loc[index, 'Size'] = float(row['Size'][:len(row['Size']) - 1])

    # convert Price to float
    df.loc[index, 'Price'] = float(row['Price'][len(row['Price']) - 1])

    # I don't know how to deal with the multiple attribute ...orz
    # df.loc[index, 'Genres'].split(';')

print (df)

