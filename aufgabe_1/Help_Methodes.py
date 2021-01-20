with open('data(2).csv', 'r') as t1, open('fileSubmission222.csv', 'r') as t2:
    fileone = t1.readlines()
    filetwo = t2.readlines()

count = 0
with open('update.csv', 'w') as outFile:
    for line in filetwo:
        if line not in fileone:
            count +=1
            outFile.write(line)
    print(count)

import pandas as pd
df = pd.read_csv('data(2).csv')
df['Predicted'].value_counts()

import pandas as pd


def Load_strip_train(number):
    df = pd.read_csv('/Users/othx30/data/train/strip_' + str(number) + '_train.csv', sep=',')
    df['r'] = df['r'].fillna(df['r'].mean())


for i in range(1, 24):
    print('dataSet ' + str(i) + ' preparing')
    df = pd.read_csv('/Users/othx30/data/train/strip_' + str(i) + '_train.csv', sep=',')
    print(df['near'].value_counts())
