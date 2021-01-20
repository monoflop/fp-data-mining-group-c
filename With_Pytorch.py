import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
import  seaborn as sns
import numpy as np
from numpy import save
from numpy import load
from numpy import asarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn import datasets


def Load_strip_train(number):
    strip_train = pd.read_csv('/Users/othx30/data/train/strip_' + str(number) + '_train.csv', sep=',')
    strip_train['r'] = strip_train['r'].fillna(strip_train['r'].mean())
    # strip_train = strip_train.groupby(['frame_number', 'run_number'])
    parameters = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']
    counter = 0
    for i, row in strip_train.groupby(['frame_number', 'run_number']):
        if (counter == 0):
            # determine number of nodes
            length = len(row.filter(items=parameters))
            print("Length is: %s" % length)
            # determine number of frames
            size = round(len(strip_train) / length)
            print("Size is: %s" % size)
            # generate 2D-numpy-Array+

            trainingData = np.zeros((size, length * len(parameters)))
            trainingLabels = np.zeros((size, 1))

        trainingData[counter,] = np.concatenate(row.filter(items=parameters).to_numpy())

        trainingLabels[counter,] = np.concatenate(row.filter(items=['near']).to_numpy())[0]

        counter += 1

    X_train, X_test, Y_train, Y_test = train_test_split(trainingData, trainingLabels, test_size=0.5, random_state=1234)

    # Load testData
    strip_test = pd.read_csv("data/test/strip_" + str(number) + "_test_no_labels.csv", sep=',')
    strip_test['r'] = strip_test['r'].fillna(strip_test['r'].mean())
    # strip_train = strip_train.groupby(['frame_number', 'run_number'])
    parameters = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']
    counter = 0
    for i, row in strip_test.groupby(['frame_number']):
        if (counter == 0):
            # determine number of nodes
            length = len(row.filter(items=parameters))
            print("Length is: %s" % length)
            # determine number of frames
            size = round(len(strip_test) / length)
            print("Size is: %s" % size)
            # generate 2D-numpy-Array+

            testData = np.zeros((size, length * len(parameters)))

        testData[counter,] = np.concatenate(row.filter(items=parameters).to_numpy())

        counter += 1

    # Scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    testData = sc.fit_transform(testData)
    testData = sc.transform(testData)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    Y_train = torch.from_numpy(Y_train.astype(np.float32))
    Y_test = torch.from_numpy(Y_test.astype(np.float32))

    Y_train = Y_train.view(Y_train.shape[0], 1)
    Y_test = Y_test.view(Y_test.shape[0], 1)

    testData = torch.from_numpy(testData.astype(np.float32))

    return [X_train, X_test, Y_train, Y_test, testData]


def main():
    print("-----------------------------------------")
    print("TU-Dortmund fp-data-mining-group-c 2020")
    print("Tool to create benchmark results")
    print("-----------------------------------------")


class NeuralNet(nn.Module):
    def __init__(self, n_features, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.n_features = n_features
        self.l1 = nn.Linear(n_features, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        y_pred = torch.sigmoid(out)

        return y_pred


hidden_size = 50
num_class = 1
# 2) Loss and optimizer
num_epochs = 10
learning_rate = 0.01

for x in range(1, 24):

    print("DataSet " + str(x) + " | Preparing", end='', flush=True)

    X_train, X_test, Y_train, Y_test, testData = Load_strip_train(x)
    n_samples, n_features = X_train.shape
    model = NeuralNet(n_features, hidden_size, num_class)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 3) Training loop
    for epoch in range(num_epochs):
        # Forward pass and loss
        # X_train = X_train.reshape(-1,n_features)
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        # if epoch % 1 == 0:
        # print(f'epoch {epoch+1}: loss = {loss:.3f}')
    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
        print(f'accuracy: {acc.item():.4f}')
    with torch.no_grad():
        y_pr = model(testData)
        print("->done", flush=True)

        f = open("solution.csv", "a")
        if x == 1:
            f = open("solution.csv", "w")
            f.write("Id,Predicted\n")

        count = 3412 * (x - 1)
        for a in y_pr:
            f.write(str(count))
            f.write(",")
            f.write(str(int(a)))
            f.write("\n")
            count = count + 1
        f.close()

