from hashlib import new
import numpy as np
from keras.models import load_model
import pandas as pd
from math import log

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
    preds = -model.predict(sampleSet)
    preds.sort()
    preds = -preds
    if method == 'pd':
        matrix = [pd(i) for i in preds]
    elif method == 'pv':
        matrix = [pv(i) for i in preds]
    else:
        matrix = [pe(i) for i in preds]
    return matrix
    

#sort(Complete_Model,x_train)
    




            

