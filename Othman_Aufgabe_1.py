
# from imblearn.over_sampling import SMOTE
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
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
# svm
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# neural network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import  make_classification


def Load_strip_train(number):
    if number == 1 or number == 22:
        # Read target dataset
        strip_train = pd.read_csv('/Users/othx30/data/train/strip_' + str(number) + '_train.csv', sep=',')

        # Drop first 15 rows
        strip_train = strip_train.iloc[1500:]

        # Read data frame with near value of 1.0
        strip_trainNear = pd.read_csv('/Users/othx30/data/train/strip_2_train.csv', sep=',')
        strip_trainNear = strip_trainNear.iloc[:1500]

        # Insert first frame into target dataset
        strip_train = pd.concat([strip_trainNear, strip_train]).reset_index(drop=True)
        #print(strip_train['near'].head(30))
    elif number == 23:
        # Read target dataset
        strip_train = pd.read_csv('/Users/othx30/data/train/strip_' + str(number) + '_train.csv', sep=',')

        # Drop first 15 rows
        strip_train = strip_train.iloc[20000:]

        # Read data frame with near value of 1.0
        strip_trainNear = pd.read_csv('/Users/othx30/data/train/strip_20_train.csv', sep=',')
        strip_trainNear = strip_trainNear.iloc[:20000]

        # Insert first frame into target dataset
        strip_train = pd.concat([strip_trainNear, strip_train]).reset_index(drop=True)
        #print(strip_train['near' == 1].head(10))
        print(strip_train['near'].value_counts())


    else :

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
    strip_test = pd.read_csv('/Users/othx30/data/test/strip_' + str(number) + '_test_no_labels.csv', sep=',')
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
        testData = np.empty((size, length * len(parameters)))
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

        # Generate datasets
       # trainingData , trainingLabels = make_classification(n_classes= 2 , weights=[0.1,0.9])

        # transform the dataset
        oversample = SMOTE(k_neighbors=2)
        Smote_X_train1, Smote_Y_train1 = oversample.fit_resample(trainingData, trainingLabels)

        #Split the data
        X_train1, X_test1, Y_train1, Y_test1 = train_test_split(trainingData, trainingLabels, test_size=0.30,
                                                                random_state=42)


        pca = PCA(n_components=4, svd_solver='auto')

        steps = [('pca', pca),
                 ('scalar', StandardScaler()),
                 ('poly', PolynomialFeatures(degree=2)),
                 ('model', RandomForestClassifier(criterion='gini', random_state=42, n_jobs=2))
                 ]

        pipeline = Pipeline(steps)
        ##################
        # Training
        # Random Forest
        print("->training", end='', flush=True)
        # forest = RandomForestClassifier(n_estimators=200, random_state=0)
        # forest.fit(X_train1, Y_train1)
        pipeline.fit(Smote_X_train1, Smote_Y_train1)

        print('Training score: {}'.format(pipeline.score(Smote_X_train1, Smote_Y_train1)))
        print('Test score: {}'.format(pipeline.score(Smote_X_train1, Smote_Y_train1)))

        ##################
        # Prediction
        print("->predicting", end='', flush=True)
        prediction = pipeline.predict(X_test1)

        print("Accuracy " + str(pipeline.score(X_test1, Y_test1)))
        print(classification_report(Y_test1, prediction))

        # Calculating Area Under the Curve :

        fprValue2, tprValue2, thresholdsValue2 = roc_curve(Y_test1, prediction)
        AUCValue = auc(fprValue2, tprValue2)
        print('AUC Value  : ', AUCValue)

        # Calculatin Balanced accuracy
        balanced_Value = balanced_accuracy_score(Y_test1, prediction)
        print(' balanced accuracy : ', balanced_Value)

        forest_prediction = pipeline.predict(testData)

        print("->done", flush=True)

        f = open("fileSubmission111.csv", "a")
        if x == 1:
            f = open("fileSubmission111.csv", "w")
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