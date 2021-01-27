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
    #Datenstruktur so ändern, dass wir pro frame eine Zeile mit 60 Sensordaten erhalten
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

    #Alle frames entfernen, wo near = 0
    rawDataSet = rawDataSet[rawDataSet.near != 0]
    rawDataSet.reset_index(inplace=True)

    #print(str(len(rawDataSet.index)))

    X_train = rawDataSet.drop('vicon_x',axis = 1)
    X_train = X_train.drop('vicon_y',axis = 1)
    
    Y_train = rawDataSet[['vicon_x', 'vicon_y']]

    #Alle nicht Sensordaten entfernen
    X_train = X_train.drop('frame_number',axis = 1)
    X_train = X_train.drop('strip_id',axis = 1)
    X_train = X_train.drop('node_id',axis = 1)
    X_train = X_train.drop('timestamp',axis = 1)
    X_train = X_train.drop('run_number',axis = 1)
    X_train = X_train.drop('near',axis = 1)

    #NaN Werte normalisieren
    #Wurde schon in der Vorverarbeitung normalisiert
    X_train = X_train.fillna(X_train.mean())
    trainingData = dataToNumpy(X_train)
    
    #Die Labels Y_train auch auf jede 15te Zeile reduzieren
    frames = len(X_train.index) / 15  
    trainingLabels = np.zeros((int(frames), 2), dtype=np.float64)
    for index, row in Y_train.iterrows():
        #print("index " + str(index))
        if(index % 15 == 0):
            trainingLabels[int(index / 15)][0] = row['vicon_x']
            trainingLabels[int(index / 15)][1] = row['vicon_y']
            
    #Daten einheitlich skalieren
    #StandardScaler
    #Base result 0.9430854964979385
    sc = StandardScaler()
    sc.fit(trainingData)
    trainingData = sc.transform(trainingData)

    #MinMaxScaler
    #Base result 0.9567132739159989
    #sc = MinMaxScaler()
    #sc.fit(trainingData)
    #trainingData = sc.transform(trainingData)

    return trainingData, trainingLabels

def saveModel(model, run):
    f = open("regression_model/model_" + str(run) + ".bin", "wb")
    f.write(model)
    f.close()

def main():
    print("-----------------------------------------")
    print("TU-Dortmund fp-data-mining-group-c 2020")
    print("Tool to create benchmark results")
    print("-----------------------------------------")

    #Train and predict for all sets
    for x in range(1, 24):
        
        ##################
        #Skip training for datasets
        #with only zeros. Because result is always zero.
        if x == 1 or x == 22 or x == 23:
            print("DataSet " + str(x) + " | Skip zero dataset")
            #ratioSum = ratioSum + 0
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
        
        ##################
        #Training
        #Random Forest
        print("->training", end = '', flush=True)
        model = RandomForestRegressor(n_estimators=200, random_state = 0)
        model.fit(trainingData, trainingLabels)

        #Save model
        print("->saving")
        s = pickle.dumps(model)
        saveModel(s, x)

        #MLP
        #mlpc=MLPClassifier(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000)
        #mlpc.fit(X_train1, Y_train1)


        #mlp = MLPClassifier(max_iter=200, random_state = 0)
        #parameter_space = {
        #    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        #    'activation': ['tanh', 'relu'],
        #    'solver': ['sgd', 'adam'],
        #    'alpha': [0.0001, 0.05],
        #    'learning_rate': ['constant','adaptive'],
        #}
        #clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        #clf.fit(X_train1, Y_train1)


if __name__ == "__main__":
    main()