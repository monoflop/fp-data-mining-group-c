#Abhängigkeiten
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle

#svm
from sklearn.svm import SVC
from sklearn import svm

#neural network
from sklearn.model_selection import GridSearchCV
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

    return testData

def readModel(run):
    f = open("classification_model/model_" + str(run) + ".bin", "rb")
    model = pickle.load(f)
    f.close()
    return model

def main():
    print("-----------------------------------------")
    print("TU-Dortmund fp-data-mining-group-c 2020")
    print("Tool to estimate position")
    print("-----------------------------------------")

    ##################
    #Load test data
    testDataSets = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

    print("Loading test data into memory")
    for x in range(1, 24):
        #Skip test for datasets
        #with only zeros. Because result is always zero.
        if x == 1 or x == 22 or x == 23:
            print("DataSet " + str(x) + " | Skip zero dataset")
            continue

        print("DataSet " + str(x) + " | Preparing", end = '', flush=True)

        #Numpy array caching
        testData = None
        cacheFile = "cache/test_" + str(x) + ".npy"
        if os.path.exists(cacheFile):
            testData = load(cacheFile)
        else:
            testData = prepareTestData(x)
            save(cacheFile, testData)

        #Store data in array
        testDataSets[x] = testData

        print("->loaded")

    ##################
    #Load classification models
    testClassificationModels = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

    print("Loading classification models")
    for x in range(1, 24):
        #Skip test for datasets
        #with only zeros. Because result is always zero.
        if x == 1 or x == 22 or x == 23:
            print("CModel " + str(x) + " | Skip zero model")
            continue
        
        testClassificationModels[x] = readModel(x)
        print("CModel " + str(x) + " | Loaded model")


    ##################
    #Iterate over all frames
    for frame in range(0, 3412):

        #Prediction is stored as an array
        #each entry represents the prediction of one strip
        framePrediction = np.zeros(23, dtype=np.int32)

        #Iterate over all strips
        for strip in range(1, 24):
            if strip == 1 or strip == 22 or strip == 23:
                framePrediction[strip - 1] = 0
                continue

            #Get model for target strip
            targetModel = testClassificationModels[strip]

            #Get frame for target strip
            targetFrameData = testDataSets[strip][frame]

            #Create prediction for frame
            prediction = targetModel.predict([targetFrameData])
            framePrediction[strip - 1] = prediction[0]
        
        print("Predicted for frame {:04d}".format(frame) + " \t" + str(framePrediction))

    



if __name__ == "__main__":
    main()