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
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def dataToNumpy(dataSet):
    #Datenstruktur so ändern, dass wir pro frame eine Zeile mit 60 Sensordaten erhalten
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
        #trainingData[frame_number][node_index * 10 + 6] = row['mx']
        #trainingData[frame_number][node_index * 10 + 7] = row['my']
        #trainingData[frame_number][node_index * 10 + 8] = row['mz']
        #trainingData[frame_number][node_index * 10 + 9] = row['r']

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

def kaggle_score(y_test, predictions):
    true_positives = 0
    true_negatives = 0
    
    false_positives = 0
    false_negatives = 0
    
    positives = 0
    negatives = 0
    
    for i in range(0,len(y_test)):
        if y_test[i] == predictions[i]:
           if y_test[i] == 0:
               true_negatives+=1
               negatives+=1
           else:
               true_positives+=1
               positives+=1
        else:
            if y_test[i] == 0:
               false_negatives+=1
               negatives+=1
            else:
               false_positives+=1
               positives+=1
    
    #First and last tow datasets produce only 0
    #So 
    
    if positives == 0:
        true_positive_ratio = 0
    else:
        true_positive_ratio = true_positives / positives
        
    if negatives == 0:
        true_negative_ratio = 0
    else:
        true_negative_ratio = true_negatives / negatives
        
    ratio = (true_positive_ratio+true_negative_ratio)/2
    print("True Positive Ratio: ",true_positive_ratio)
    print("True Negative Ratio: ",true_negative_ratio)
    print("True Ratio: ", ratio)
    return ratio

def main():
    print("-----------------------------------------")
    print("TU-Dortmund fp-data-mining-group-c 2020")
    print("Tool to create benchmark results")
    print("-----------------------------------------")

    ratioSum = 0

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
        
        ##################
        #Training
        X_train1, X_test1, Y_train1, Y_test1 = train_test_split(trainingData, trainingLabels, test_size=0.20, random_state=42)

        #Random Forest
        print("->training", end = '', flush=True)
        forest = RandomForestClassifier(n_estimators=200, random_state = 0)
        forest.fit(X_train1, Y_train1)

        #MLP
        mlpc=MLPClassifier()
        mlpc.fit(X_train1, Y_train1)


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

        ##################
        #Prediction
        print("Predicting")
        prediction = mlpc.predict(X_test1)
        #print(classification_report(Y_test1, prediction))
        score = kaggle_score(Y_test1, prediction)
        print("Kaggle_score " + str(score))
        ratioSum = ratioSum + score

    ratioSum = ratioSum / 23
    print("Final score " + str(ratioSum))


if __name__ == "__main__":
    main()