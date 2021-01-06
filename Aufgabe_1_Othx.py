
#from imblearn.over_sampling import SMOTE
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#svm
from sklearn.svm import SVC
from sklearn import svm

#neural network
from sklearn.neural_network import MLPClassifier


def Load_strip_train(number):
    strip_train = pd.read_csv('/Users/othx30/data/train/strip_' + str(number) + '_train.csv', sep=',')
    strip_train['r'] = strip_train['r'].fillna(strip_train['r'].mean())
    # strip_train = strip_train.groupby(['frame_number', 'run_number'])
    parameters = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']
    counter = 0
    for i, row in strip_train.groupby(['frame_number', 'run_number']):
        if (counter == 0):
            # determine number of nodes
            length = len(row.filter(items=parameters).to_numpy())
            print("Length is: %s" % length)
            # determine number of frames
            size = round(len(strip_train) / length)
            print("Size is: %s" % size)
            # generate 2D-numpy-Array
            trainingData = np.zeros((size, length * len(parameters)))
            trainingLabels = np.zeros((size, 1))

        trainingData[counter,] = np.concatenate(row.filter(items=parameters).to_numpy())

        trainingLabels[counter,] = np.concatenate(row.filter(items=['near']).to_numpy())[0]

        counter += 1

    return [trainingData, trainingLabels]


def Load_strip_test(number):
    strip_test = pd.read_csv("/Users/othx30/data/test/strip_" + str(number) + "_test_no_labels.csv", sep=',')
    strip_test = strip_test.fillna(strip_test.mean())
    parameters = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']
    counter = 0
    for i, row in strip_test.groupby('frame_number'):
        if (counter == 0):
            # determine number of nodes
            length = len(row.filter(items=parameters).to_numpy())
            print("Length is: %s" % length)
            # determine number of frames
            size = round(len(strip_test) / length)
            print("Length is: %s" % size)
        testData = np.empty((size, length*len(parameters)))
        testData[counter,] = np.concatenate(row.filter(items=parameters).to_numpy())
        counter += 1

    return testData


def main():
    print("-----------------------------------------")
    print("TU-Dortmund fp-data-mining-group-c 2020")
    print("Tool to create benchmark results")
    print("-----------------------------------------")



    # Train and predict for all sets
    for x in range(1, 24):

        print("DataSet " + str(x) + " | Preparing", end='', flush=True)

        ##################
        # Numpy array caching
        # Training data
        trainingData = None
        trainingLabels = None
        cacheTrainingFile = "cache/train_" + str(x) + ".npy"
        cacheTrainingLabelFile = "cache/train_label_" + str(x) + ".npy"
        if os.path.exists(cacheTrainingFile):
            trainingData = load(cacheTrainingFile)
            trainingLabels = load(cacheTrainingLabelFile)
        else:
            trainingData, trainingLabels = Load_strip_train(x)
            save(cacheTrainingFile, trainingData)
            save(cacheTrainingLabelFile, trainingLabels)

        # Test data
        testData = None
        cacheFile = "cache/test_" + str(x) + ".npy"
        if os.path.exists(cacheFile):
            testData = load(cacheFile, allow_pickle=True)
        else:
            testData = Load_strip_test(x)
            save(cacheFile, testData)

        X_train1, X_test1, Y_train1, Y_test1 = train_test_split(trainingData, trainingLabels, test_size=0.20, random_state=42)

        pca = PCA()

        steps = [('pca',pca),
            ('scalar', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('model', RandomForestClassifier(n_estimators=100,max_depth=1, random_state=42))
        ]

        pipeline = Pipeline(steps)
        ##################
        # Training
        # Random Forest
        print("->training", end='', flush=True)
        #forest = RandomForestClassifier(n_estimators=200, random_state=0)
        #forest.fit(X_train1, Y_train1)
        param_grid = {'pca__n_components': [1, 2, 4, 7, 9],
                      'model__criterion':['gini','entropy'],
                      'poly__degree':[2,3,4,5],}

        search = GridSearchCV(pipeline, param_grid, iid=False, cv=5, return_train_score=False)


        search.fit(X_train1, Y_train1)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        print('Training score: {}'.format(search.score(X_train1, Y_train1)))
        print('Test score: {}'.format(search.score(X_test1, Y_test1)))

        ##################
        # Prediction
        prediction = search.predict(X_test1)
        print("Accuracy " + str(search.score(X_test1, Y_test1)))
        print(classification_report(Y_test1, prediction))
        print("->predicting", end='', flush=True)
        forest_prediction = search.predict(testData)





        print("->done", flush=True)

        f = open("fileSubmission222.csv", "a")
        if x == 1:
            f = open("fileSubmission222.csv", "w")
            f.write("Id,Predicted\n")

        count = 3412 * (x - 1)
        for a in forest_prediction:
            f.write(str(count))
            f.write(",")
            f.write(str(int(a)))
            f.write("\n")
            count = count + 1
        f.close()
        # break
if __name__ == "__main__":
    main()
