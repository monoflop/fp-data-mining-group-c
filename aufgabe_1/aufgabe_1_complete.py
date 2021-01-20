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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

#svm
from sklearn.svm import SVC
from sklearn import svm

#neural network
from sklearn.neural_network import MLPClassifier

def dataToNumpy(dataSet):
    #Datenstruktur so ändern, dass wir pro frame eine Zeile mit 150 Sensordaten erhalten
    #Anzahl Zeilen / 15 = Frames
    frames = len(dataSet.index) / 15
    trainingData = np.zeros((int(frames),150), dtype=np.float64)
    frame_number = 0
    node_index = 0
    for i, row in dataSet.iterrows():
        trainingData[frame_number][node_index * 10 + 0] = row['ax']
        trainingData[frame_number][node_index * 10 + 1] = row['ay']
        trainingData[frame_number][node_index * 10 + 2] = row['az']
        trainingData[frame_number][node_index * 10 + 3] = row['gx']
        trainingData[frame_number][node_index * 10 + 4] = row['gy']
        trainingData[frame_number][node_index * 10 + 5] = row['gz']
        trainingData[frame_number][node_index * 10 + 6] = row['mx']
        trainingData[frame_number][node_index * 10 + 7] = row['my']
        trainingData[frame_number][node_index * 10 + 8] = row['mz']
        trainingData[frame_number][node_index * 10 + 9] = row['r']
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
            
    #Daten einheitlich skalieren
    #StandardScaler
    #Base result 0.9430854964979385
    scaler = StandardScaler()
    scaler.fit(trainingData)
    trainingData = scaler.transform(trainingData)

    #MinMaxScaler
    #Base result 0.9567132739159989
    #scaler = MinMaxScaler()
    #scaler.fit(trainingData)
    #trainingData = scaler.transform(trainingData)

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

    #Daten einheitlich skalieren
    #StandardScaler
    scaler = StandardScaler()
    scaler.fit(testData)
    testData = scaler.transform(testData)

    #MinMaxScaler
    #scaler = MinMaxScaler()
    #scaler.fit(testData)
    #testData = scaler.transform(testData)

    return testData

def savePrediction(prediction, run):
    f = open("data.csv", "a")
    if run == 1:
        f = open("data.csv", "w")
        f.write("Id,Predicted\n")

    count = 3412 * (run - 1)
    for a in prediction:
        f.write(str(count))
        f.write(",")
        f.write(str(int(a)))
        f.write("\n")
        count = count + 1 
    f.close()

def saveModel(model, run):
    f = open("model/model_" + str(run) + ".bin", "wb")
    f.write(model)
    f.close()

def main():
    print("-----------------------------------------")
    print("TU-Dortmund fp-data-mining-group-c 2020")
    print("Tool to create benchmark results")
    print("-----------------------------------------")

    #Train and predict for all sets
    for x in range(1, 24):

        #TODO test
        if x == 1 or x == 22 or x == 23:
            print("DataSet " + str(x) + " | Skip zero dataset")
            #Write empty prediction
            prediction = [0] * 3412
            savePrediction(prediction, x)
            continue

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
        model = RandomForestClassifier(n_estimators=200, random_state = 0)
        model.fit(trainingData, trainingLabels)

        #SVM
        #model = svm.SVC()
        #model.fit(trainingData, trainingLabels)

        #Neural network
        #model=MLPClassifier()
        #model.fit(trainingData, trainingLabels)

        ##################
        #Prediction
        print("->predicting", end = '', flush=True)
        prediction = model.predict(testData)

        #Save model
        print("->saving", end = '', flush=True)
        s = pickle.dumps(model)
        saveModel(s, x)

        print("->done", flush=True)

        savePrediction(prediction, x)

if __name__ == "__main__":
    main()