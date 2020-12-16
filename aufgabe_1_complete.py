#Abhängigkeiten
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#svm
from sklearn.svm import SVC
from sklearn import svm

#neural network
from sklearn.neural_network import MLPClassifier

def dataToNumpy(dataSet):
    #Datenstruktur so ändern, dass wir pro frame eine Zeile mit 150 Sensordaten erhalten
    #Anzahl Zeilen / 15 = Frames
    frames = len(dataSet.index) / 15
    trainingData = np.zeros((int(frames),60), dtype=np.float64)
    frame_number = 0
    node_index = 0
    for i, row in dataSet.iterrows():
        #trainingData[frame_number][node_index * 10 + 0] = row['ax']
        #trainingData[frame_number][node_index * 10 + 1] = row['ay']
        #trainingData[frame_number][node_index * 10 + 2] = row['az']
        #trainingData[frame_number][node_index * 10 + 3] = row['gx']
        #trainingData[frame_number][node_index * 10 + 4] = row['gy']
        #trainingData[frame_number][node_index * 10 + 5] = row['gz']
        trainingData[frame_number][node_index * 4 + 0] = row['mx']
        trainingData[frame_number][node_index * 4 + 1] = row['my']
        trainingData[frame_number][node_index * 4 + 2] = row['mz']
        trainingData[frame_number][node_index * 4 + 3] = row['r']
        node_index = node_index + 1
        if(node_index >= 15):
            frame_number = frame_number + 1
            node_index = 0
    return trainingData
    
def prepareTrainingData(dataIndex):
    ##################
    #Trainingsdaten
    #Label aus Datensatz entfernen, damit wir einen Label-Satz und einen Feature-Satz erhalten.
    rawDataSet = pd.read_csv("data/train/strip_" + str(dataIndex) + "_train.csv", sep=',')
    X_train = rawDataSet.drop('near',axis = 1)
    Y_train = rawDataSet['near']

    #Alle nicht Sensordaten entfernen
    X_train = X_train.drop('frame_number',axis = 1)
    X_train = X_train.drop('strip_id',axis = 1)
    X_train = X_train.drop('node_id',axis = 1)
    X_train = X_train.drop('timestamp',axis = 1)
    X_train = X_train.drop('run_number',axis = 1)
    X_train = X_train.drop('vicon_x',axis = 1)
    X_train = X_train.drop('vicon_y',axis = 1)

    #NaN Werte normalisieren
    #Wurde schon in der Vorverarbeitung normalisiert
    #TODO check
    X_train = X_train.fillna(X_train.mean())
    trainingData = dataToNumpy(X_train)
    
    #Die Labels Y_train auch auf jede 15te Zeile reduzieren
    frames = len(X_train.index) / 15  
    trainingLabels = np.zeros((int(frames)), dtype=np.int64)
    for i, number in Y_train.iteritems():
        if(i % 15 == 0):
            trainingLabels[int(i / 15)] = int(number)
            
    #Daten einheitlich skalieren von z. B. 0.0 - 1.0
    sc = StandardScaler()
    sc.fit(trainingData)
    trainingData = sc.transform(trainingData)
    return trainingData, trainingLabels

def prepareTestData(dataIndex):
    ##################
    #Testdaten
    #Alle nicht Sensordaten entfernen
    rawDataSet = pd.read_csv("data/test/strip_" + str(dataIndex) + "_test_no_labels.csv", sep=',')
    X_test = rawDataSet.drop('frame_number',axis = 1)
    X_test = X_test.drop('strip_id',axis = 1)
    X_test = X_test.drop('node_id',axis = 1)
    X_test = X_test.drop('timestamp',axis = 1)

    X_test = X_test.fillna(X_test.mean())
    testData = dataToNumpy(X_test)

    #Daten einheitlich skalieren von z. B. 0.0 - 1.0
    sc = StandardScaler()
    sc.fit(testData)
    testData = sc.transform(testData)
    return testData

def main():
    print("-----------------------------------------")
    print("TU-Dortmund fp-data-mining-group-c 2020")
    print("Tool to create benchmark results")
    print("-----------------------------------------")

    #Train and predict for all sets
    for x in range(1, 24):
        print("DataSet " + str(x) + " | Preparing", end = '', flush=True)

        ##################
        #Numpy array caching
        #Training data
        trainingData = None
        trainingLabels = None
        cacheTrainingFile = "cache/train_" + str(x) + ".npy"
        cacheTrainingLabelFile = "cache/train_label_" + str(x) + ".npy"
        if os.path.exists(cacheTrainingFile):
            trainingData = load(cacheTrainingFile)
            trainingLabels = load(cacheTrainingLabelFile)
        else:
            trainingData, trainingLabels = prepareTrainingData(x)
            save(cacheTrainingFile, trainingData)
            save(cacheTrainingLabelFile, trainingLabels)

        #Test data
        testData = None
        cacheFile = "cache/test_" + str(x) + ".npy"
        if os.path.exists(cacheFile):
            testData = load(cacheFile)
        else:
            testData = prepareTestData(x)
            save(cacheFile, testData)
        
        ##################
        #Training
        #Random Forest
        print("->training", end = '', flush=True)
        #forest = RandomForestClassifier(n_estimators=200, random_state = 0)
        #forest.fit(trainingData, trainingLabels)

        #SVM
        #SVM does not work on the first set because it only has zero labels
        #clf = svm.SVC()
        #clf.fit(trainingData, trainingLabels)

        #Neural network
        mlpc=MLPClassifier()
        mlpc.fit(trainingData, trainingLabels)

        ##################
        #Prediction
        print("->predicting", end = '', flush=True)
        prediction = mlpc.predict(testData)

        print("->done", flush=True)

        f = open("data.csv", "a")
        if x == 1:
            f = open("data.csv", "w")
            f.write("Id,Predicted\n")

        count = 3412 * (x - 1)
        for a in prediction:
            f.write(str(count))
            f.write(",")
            f.write(str(int(a)))
            f.write("\n")
            count = count + 1 
        f.close()
        #break

if __name__ == "__main__":
    main()