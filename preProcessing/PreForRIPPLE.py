from genericpath import sameopenfile
from hashlib import new
from random import sample
import numpy as np
from keras.models import load_model,Model
import pandas as pd
from math import log
import argparse
import os
# augmentSize = 10
# maxDistance = 0.1

#Complete_Model = load_model("../data/mnist_adv/mnist_adv.h5")
#inputShape = np.array(Complete_Model.input_shape)
#for i in range(len(inputShape)):
#    if (type(inputShape[i]) == type(None)):
#        inputShape[i] = -1
#y_train = np.array(pd.read_csv("../data/mnist_adv/train_labels.csv",header=None))
#x_train = np.array(pd.read_csv("../data/mnist_adv/train_samples.csv",header=None)).reshape(inputShape)

def pd(pred):
    result = 0.0
    for i in range(len(pred)-1):
        result += (pred[i] - pred[i+1])/float(i+1)
    return result

def pv(pred):
    result = 0.0
    avg = 0.0
    for i in pred:
        avg += i
    avg /= len(pred)
    for i in range(len(pred)):
        result += (pred[i] - avg)**2
    return result

def pe(pred):
    result = 0.0
    for i in range(len(pred)):
        result += -pred[i]*log(pred[i])
    return result


def sort(model,sampleSet,method="pd"):
    inputShape = np.array(model.input_shape)
    for i in range(len(inputShape)):
        if (type(inputShape[i]) == type(None)):
            inputShape[i] = -1
    #print(inputShape)
    preds = -model.predict(sampleSet.reshape(inputShape))
    preds.sort()
    preds = -preds
    if method == 'pd':
        matrix = [pd(i) for i in preds]
    elif method == 'pv':
        matrix = [pv(i) for i in preds]
    else:
        matrix = [pe(i) for i in preds]
    return matrix

def mySort(model,sampleSet,labelSet):
    #InputShape
    inputShape = np.array(model.input_shape)
    for i in range(len(inputShape)):
        if (type(inputShape[i]) == type(None)):
            inputShape[i] = -1
    
    #w and ConfVect
    for i in range(-1, -1000, -1):
	    if len(model.get_layer(index=i).get_weights()) == 2:
		    break
    w = model.get_layer(index=i).get_weights()[0]
    b = model.get_layer(index=i).get_weights()[1]
    newModel = Model(inputs=model.input, outputs=model.layers[i-1].output)
    preds = newModel.predict(sampleSet.reshape(inputShape))
    preds = np.concatenate((preds,np.ones(shape=(len(preds),1))),axis=1)
    ConfVect = np.dot(preds,np.concatenate((w,b.reshape(1,-1))))

    w = w.transpose()
    CateNum = len(w)
    Diff = [[0 for j in range(CateNum)] for i in range(CateNum)]
    for i in range(CateNum):
        for j in range(CateNum):
            if (i==j):
                continue
            else:
                Diff[i][j] = np.linalg.norm(w[i]-w[j])
    
    Metrix = []
    for c,l in zip(ConfVect,labelSet):
        #R_c for positive Samples
        if np.argmax(c) == l:
            R_c = 100000
            for j in range(CateNum):
                if j==l:
                    continue
                else:
                    tempR = (c[l]-c[j])/Diff[l][j]
                    if (tempR < R_c):
                        R_c = tempR
            Metrix.append(R_c)
        else:
            R_c = -1
            for j in range(CateNum):
                if j==l:
                    continue
                else:
                    tempR = (c[j]-c[l])/Diff[j][l]
                    if (tempR > R_c):
                        R_c = tempR
            Metrix.append(R_c)
    return Metrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--network")
    parser.add_argument("-x",'--trainData')
    parser.add_argument("-y",'--trainLabels')
    parser.add_argument("-xa",'--trainDatax')
    parser.add_argument("-ya",'--trainLabelsx')
    #parser.add_argument("-s","--size",type=int)
    parser.add_argument("-a",'--positiveRadio', type=float)
    parser.add_argument("-b",'--negativeRadio',type=float)
    parser.add_argument("-d",'--outputData')
    parser.add_argument("-l",'--outputLabels')
    args = parser.parse_args()

    model = load_model(args.network)
    inputShape = np.array(model.input_shape)
    for i in range(len(inputShape)):
        if (type(inputShape[i]) == type(None)):
            inputShape[i] = -1
    labelSet = np.load(args.trainLabels)#[:args.size]
    sampleSet = np.load(args.trainData)#[:args.size]
    sampleSet = sampleSet.reshape(inputShape)
    #print(sampleSet.shape)
    matrix = mySort(model,sampleSet,labelSet)
    
    #thresholdL = sorted(matrix)[int(len(matrix)*0.5)]
    #thresholdR = sorted(matrix)[int(len(matrix)*0.6)]
    pred = np.argmax( model.predict(sampleSet.reshape(inputShape)),axis=1)
    print(np.sum(np.equal(pred,labelSet))/len(labelSet))
    #print(sorted(matrix))
    matrixC = []
    for m, p, l in zip(matrix,pred,labelSet):
      if (p==l):
        matrixC.append(m)
    #print(len(matrixC),len(matrixW))
    labelSetx = np.load(args.trainLabelsx)#[:args.size]
    sampleSetx = np.load(args.trainDatax)#[:args.size]
    #print(sampleSet.shape)
    matrixx = mySort(model,sampleSetx,labelSetx)
    
    #thresholdL = sorted(matrix)[int(len(matrix)*0.5)]
    #thresholdR = sorted(matrix)[int(len(matrix)*0.6)]
    predx = np.argmax( model.predict(sampleSetx.reshape(inputShape)),axis=1)
    print(np.sum(np.equal(predx,labelSetx))/len(labelSetx))
    print((sampleSet.shape,sampleSetx.shape))
    #print(sorted(matrix))
    matrixW = []
    for m, p, l in zip(matrixx,predx,labelSetx):
      if (p!=l):
        matrixW.append(m)

    thresholdCL = sorted(matrixC)[int(len(matrixC)/2*(1-args.positiveRadio))]
    thresholdCU = sorted(matrixC)[int(len(matrixC)/2*(1+args.positiveRadio))-1]
    thresholdWL = sorted(matrixW)[int(len(matrixW)/2*(1-args.negativeRadio))]
    thresholdWU = sorted(matrixW)[int(len(matrixW)/2*(1+args.negativeRadio))-1]
    #thresholdW = sorted(matrixW)[int(len(matrixW)*args.negativeRadio)-1]
    filteredSamples = []
    filteredLabels = []
    cntP = 0
    cntN = 0
    for i,sample in enumerate(sampleSet):
        #print(sample.shape)
        if (pred[i] == labelSet[i] and matrix[i] <= thresholdCU and matrix[i] >= thresholdCL and cntP < int(len(matrixC)*args.positiveRadio) ):
            cntP += 1
        else:
            continue
        filteredSamples.append(sample)
        filteredLabels.append(labelSet[i])
    for i,sample in enumerate(sampleSetx):
        #print(sample.shape)
        if (predx[i] != labelSetx[i] and matrixx[i] <= thresholdWU and matrixx[i] >= thresholdWL and cntN < int(len(matrixW)*args.negativeRadio) ):
            cntN += 1
        else:
            continue
        filteredSamples.append(sample)
        filteredLabels.append(labelSetx[i])
    filteredSamples = np.array(filteredSamples)
    filteredLabels = np.array(filteredLabels)
    print((filteredSamples.shape,filteredLabels.shape))
    #print(f"a:{args.positiveRadio},b:{args.negativeRadio},originLen:{len(sampleSet)},afterLen:{len(filteredLabels)}")
    np.save(args.outputData,filteredSamples)
    np.save(args.outputLabels,filteredLabels)
    pred = np.argmax(model.predict(filteredSamples.reshape(inputShape)),axis=1)
    print(np.sum(np.equal(pred,filteredLabels))/len(filteredLabels))
    


    


#sort(model,x_train)
    




            

