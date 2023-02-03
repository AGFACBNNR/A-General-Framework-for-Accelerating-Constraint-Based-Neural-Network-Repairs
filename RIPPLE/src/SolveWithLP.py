import gurobipy
import time
import numpy as np
from keras.models import load_model
from keras.models import Model
import resource
import argparse
import gc

soft,hard = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (64*1024*1024*1024, hard))

epsilon = 0.00001
tolerance = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("-n","--network",help="The network to repair")
parser.add_argument("-log","--log_file",help = "File to record the result")
parser.add_argument("-r","--patch_file",help = "File to record the patch")
parser.add_argument("-b","--buggy_examples",help = "The buggy examples")
parser.add_argument("-bl","--buggy_labels",help = "The buggy examples")
parser.add_argument("-normb","--normalizeb",type=bool,default = False)
parser.add_argument("-p","--positive_examples",help = "The buggy examples")
parser.add_argument("-pl","--positive_labels",help = "The buggy examples")
parser.add_argument("-normp","--normalizep",type=bool,default = False)
#parser.add_argument("-t","--tolerance")
args = parser.parse_args()

#tolerance = float(args.tolerance)

#load_model
Complete_Model = load_model(args.network)
for i in range(-1, -1000, -1):
	print(i)
	if len(Complete_Model.get_layer(index=i).get_weights()) == 2:
		break
w = Complete_Model.get_layer(index=i).get_weights()[0]
b = Complete_Model.get_layer(index=i).get_weights()[1]
w = np.concatenate((w,b.reshape(1,-1)))
newModel = Model(inputs=Complete_Model.input, outputs=Complete_Model.layers[i-1].output)

inputShape = np.array(Complete_Model.input_shape)
for i in range(len(inputShape)):
        if (type(inputShape[i]) == type(None)):
            inputShape[i] = -1

#load_data
x_train = np.load(args.positive_examples).reshape(inputShape)
#x_train = x_train.reshape(x_train.shape[0],-1).astype("float32")
if args.normalizep:
	x_train /= 255.0
y_train = np.load(args.positive_labels)
f_train = newModel.predict(x_train)
f_train = np.concatenate((f_train,np.ones(shape=(len(f_train),1))),axis=1)

x_buggy = np.load(args.buggy_examples).reshape(inputShape)
#x_buggy = x_buggy.reshape(x_buggy.shape[0],-1).astype("float32")
y_buggy = np.load(args.buggy_labels)
if args.normalizeb:
	x_buggy /= 255.0
f_buggy = newModel.predict(x_buggy)
f_buggy = np.concatenate((f_buggy,np.ones(shape=(len(f_buggy),1))),axis=1)

#Split
pred_buggy = np.argmax(np.dot(f_buggy,w),axis=1)
negData=[]
negLabels=[]
for i in range(len(pred_buggy)):
    norm = np.linalg.norm(f_buggy[i])
    if pred_buggy[i] != y_buggy[i]:
        negData.append(f_buggy[i]/norm)
        negLabels.append(y_buggy[i])
negLen = len(negLabels)

nodeLen = f_buggy.shape[1]

del f_buggy
del pred_buggy
del x_buggy
del y_buggy
gc.collect()

pred_train = np.argmax(np.dot(f_train,w),axis=1)
posData=[]
posLabels=[]
posLen = 0
for i in range(len(pred_train)):
	norm = np.linalg.norm(f_train[i])
	if pred_train[i] == y_train[i]:
		posLen += 1
		posData.append(f_train[i]/norm)
		posLabels.append(y_train[i])

del f_train
del pred_train
del x_train
del y_train
gc.collect()

print(f"\n\n\n\n\n\n\n\n\n\n\nposLen : {posLen}\nnegLen : {negLen}\n\n\n\n")
start = time.time()
#Build Model
print("Building Model\n")
MODEL = gurobipy.Model()
#variables

outLen = Complete_Model.layers[-1].output.shape[1]
xLen = outLen*nodeLen
w = w.transpose().reshape((-1))
print(xLen,w.shape)
x = MODEL.addMVar(xLen,lb= w-tolerance,ub = w+tolerance)
MODEL.update()
#MODEL.setParam("Method",1)
#Constr
#Part1 OriginData HardConstr
posParam = []
for ind in range(posLen):
	cate = posLabels[ind]
	eles = posData[ind]
	for outInd in range(outLen):
		if outInd == cate:
			continue
		oneConstr = np.zeros(xLen)
		for i in range(nodeLen):
			a = cate*nodeLen+i
			oneConstr[a] = eles[i]
			oneConstr[outInd*(nodeLen)+i] = -eles[i]
		posParam.append(oneConstr)
posParam = np.array(posParam)

b = np.full((len(posParam),1),epsilon)
if len(posParam)>0:
	MODEL.addMConstr( posParam, x, '>', b)
OA = np.zeros((posParam.shape[1])) #posParam.sum(axis=0)

del b
del posParam
del posData
del posLabels
gc.collect()


for ind in range(negLen):
	cate = negLabels[ind]
	eles = negData[ind]
	for outInd in range(outLen):
		if outInd == cate:
			continue
		for i in range(nodeLen):
			OA[cate*nodeLen+i] += eles[i]
			OA[outInd*nodeLen+i] -= eles[i]

#if len(negParam)>0:
#        MODEL.addMConstr( negParam, x, '>', b)

del negLabels
del negData
gc.collect()


#OA = negLen*posParam.sum(axis=0)+posLen*negParam.sum(axis=0)
#OA = negParam.sum(axis=0)
MODEL.setObjective(OA@x,gurobipy.GRB.MAXIMIZE)

#solve
#MODEL.write("Prob.lp")
print("Solving\n")
MODEL.optimize()	

totalTime=time.time()-start
print(f"Total time: {totalTime}")

W = []
for i in range(nodeLen):
	newW = []
	for k in range(outLen):
		newW.append(x[k*nodeLen+i].x)
	W.append(newW)	
W = np.array(W)
W = W.reshape(W.shape[0],W.shape[1])
			
fout = open(args.patch_file,"w")
for i in range(outLen):
	for j in range(nodeLen):
		fout.write(f"{W[j][i]},")
	fout.write("\n")
fout.close()

fres = open(args.log_file,"a+")	
fres.write(f"{totalTime}\n")

	
	
	
	
	
	
	
	
