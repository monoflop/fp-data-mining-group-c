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

def readClassificationModel(run):
    f = open("classification_model/model_" + str(run) + ".bin", "rb")
    model = pickle.load(f)
    f.close()
    return model

def readRegressionModel(run):
    f = open("regression_model/model_" + str(run) + ".bin", "rb")
    model = pickle.load(f)
    f.close()
    return model

def savePrediction(prediction):
    f = open("data.csv", "w")
    f.write("vicon_x,vicon_y,frame_number\n")
    for frame in range(0, 3412):
        f.write("" + str(prediction[frame][0]) + ", " + str(prediction[frame][1]) + ", " + str(frame) + "\n")

    f.close()

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
        
        testClassificationModels[x] = readClassificationModel(x)
        print("CModel " + str(x) + " | Loaded model")

    ##################
    #Load regression models
    testRegressionModels = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

    print("Loading regression models")
    for x in range(1, 24):
        #Skip test for datasets
        #with only zeros. Because result is always zero.
        if x == 1 or x == 22 or x == 23:
            print("RModel " + str(x) + " | Skip zero model")
            continue
        
        testRegressionModels[x] = readRegressionModel(x)
        print("RModel " + str(x) + " | Loaded model")

    ##################
    #Iterate over all frames and create prediction
    positionPrediction = np.zeros((int(3412),2), dtype=np.float64)
    lastPositiveStrip = None
    predictedCaseOne = 0
    predictedCaseTwo = 0
    predictedCaseThree = 0
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

        print("Predicted for frame {:04d}".format(frame) + " \t" + str(framePrediction), end = '', flush=True)

        #Check which prediction case we have
        #Case 1 : Only one positive prediction
        #Case 2 : No positive prediction
        #Case 3 : Multiple positive predictions

        positiveCount = np.count_nonzero(framePrediction == 1)
        if positiveCount == 1:
            #Case 1
            #Predict position with regressor on only one strip
            print(" case 1", end = '', flush=True)
            targetStripIndices = np.where(framePrediction == 1)

            #+1 because array index 0 is strip 1
            targetStripIndex = (int(targetStripIndices[0][0])) + 1

            #Check if position is realistic
            if lastPositiveStrip is not None:
                diff = abs(lastPositiveStrip - targetStripIndex)
                if diff >= 3: 
                    #Position is unrealistic
                    positionPrediction[frame][0] = positionPrediction[frame - 1][0]
                    positionPrediction[frame][1] = positionPrediction[frame - 1][1]
                    print(" predicted " + str(positionPrediction[frame]))
                    lastPositiveStrip = None
                    predictedCaseOne = predictedCaseOne + 1
                    continue

            targetModel = testRegressionModels[targetStripIndex]
            targetFrameData = testDataSets[targetStripIndex][frame]
            prediction = targetModel.predict([targetFrameData])

            #Store prediction
            positionPrediction[frame][0] = prediction[0][0]
            positionPrediction[frame][1] = prediction[0][1]

            print(" predicted " + str(positionPrediction[frame]))

            lastPositiveStrip = targetStripIndex

            predictedCaseOne = predictedCaseOne + 1

        elif positiveCount == 0:
            #Case 2
            print(" case 2", end = '', flush=True)
            #TODO just use prev prediction

            #Predict next frame and calculate mean

            #Write zero and calculate mean later
            #positionPrediction[frame][0] = positionPrediction[frame - 1][0]
            #positionPrediction[frame][1] = positionPrediction[frame - 1][1]

            positionPrediction[frame][0] = 0
            positionPrediction[frame][1] = 0

            print(" predicted " + str(positionPrediction[frame]))

            lastPositiveStrip = None

            predictedCaseTwo = predictedCaseTwo + 1
        elif positiveCount > 1:
            #Case 3
            print(" case 3", end = '', flush=True)

            targetStripIndices = np.where(framePrediction == 1)

            #If the predictions are far apart, we only use the prediction that is closer to prev prediction
            #Sort desceding
            targetStripIndices[0][::-1].sort()
            diff = targetStripIndices[0][0] + targetStripIndices[0][0]
            for arrayIndex in range(len(targetStripIndices[0])):
                #print("index " + str(targetStripIndices[0][arrayIndex]))
                diff = diff - targetStripIndices[0][arrayIndex]

            diff = abs(diff)

            print(" diff " + str(diff), end = '', flush=True)

            #Predict for all positive results and calculate mean
            if diff <= 2:
                
                finalPrediction = np.zeros((2), dtype=np.float64)
                for arrayIndex in range(len(targetStripIndices[0])):
                    targetStripIndex = (int(targetStripIndices[0][arrayIndex])) + 1

                    targetModel = testRegressionModels[targetStripIndex]
                    targetFrameData = testDataSets[targetStripIndex][frame]
                    prediction = targetModel.predict([targetFrameData])

                    #Store prediction
                    finalPrediction[0] = finalPrediction[0] + prediction[0][0]
                    finalPrediction[1] = finalPrediction[1] + prediction[0][1]

                #Calulate mean value
                finalPrediction[0] = finalPrediction[0] / positiveCount
                finalPrediction[1] = finalPrediction[1] / positiveCount

                positionPrediction[frame][0] = finalPrediction[0]
                positionPrediction[frame][1] = finalPrediction[1]

            #Predictions are too far away, so we use the pre prediction
            else:
                #TODO caluclate closest to prev
                positionPrediction[frame][0] = positionPrediction[frame - 1][0]
                positionPrediction[frame][1] = positionPrediction[frame - 1][1]

            print(" predicted " + str(positionPrediction[frame]))

            lastPositiveStrip = None

            predictedCaseThree = predictedCaseThree + 1

    #Run over all predictions and fill zero values with mean between previous and next frame
    #TODO handle if multiple successively frames are zero (interpolate)
    for frame in range(len(positionPrediction)):
        if positionPrediction[frame][0] == 0 and positionPrediction[frame][1] == 0 and frame != 0 and frame != 3412:

            #If next frame is also zero, we just copy the prev frame
            if positionPrediction[frame + 1][0] == 0 and positionPrediction[frame + 1][1] == 0:
                positionPrediction[frame][0] = positionPrediction[frame - 1][0]
                positionPrediction[frame][1] = positionPrediction[frame - 1][1]
            else:
                positionPrediction[frame][0] = (positionPrediction[frame - 1][0] + positionPrediction[frame + 1][0]) / 2
                positionPrediction[frame][1] = (positionPrediction[frame - 1][1] + positionPrediction[frame + 1][1]) / 2

            print("Filled frame {:04d}".format(frame) + " \t with " + str(positionPrediction[frame]))


    print("Prediction result")
    print("case 1 " + str(predictedCaseOne))
    print("case 2 " + str(predictedCaseTwo))
    print("case 3 " + str(predictedCaseThree))

    savePrediction(positionPrediction)


if __name__ == "__main__":
    main()