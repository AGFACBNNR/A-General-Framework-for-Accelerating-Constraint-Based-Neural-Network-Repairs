from keras.models import Model
from keras.models import load_model
import numpy as np
import argparse

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
args = parser.parse_args()

#load model
model = load_model(args.network)
inputShape = np.array(model.input_shape)
for i in range(len(inputShape)):
        if (type(inputShape[i]) == type(None)):
            inputShape[i] = -1
for i in range(-1, -1000, -1):
	if len(model.get_layer(index=i).get_weights()) == 2:
		break
newModel = Model(inputs=model.input, outputs=model.layers[i-1].output)

#load patch
fin = open(args.patch_file)
W = []
for i in range(10):
	line = fin.readline()
	eles = line.strip().split(',')[:-1]
	eles = list(map(float,eles))
	W.append(eles)
W=np.array(W)

#load_data
x_train = np.load(args.positive_examples).reshape(inputShape) #.reshape((-1,28,28,1))
if args.normalizep:
	x_train /= 255.0
y_train = np.load(args.positive_labels)
f_train = newModel.predict(x_train)
f_train = np.concatenate((f_train,np.ones(shape=(len(f_train),1))),axis=1)

x_buggy = np.load(args.buggy_examples).reshape(inputShape) #.reshape((-1,28,28,1))
y_buggy = np.load(args.buggy_labels)
if args.normalizeb:
	x_buggy /= 255.0
f_buggy = newModel.predict(x_buggy)
f_buggy = np.concatenate((f_buggy,np.ones(shape=(len(f_buggy),1))),axis=1)

accuracy_pos_origin = np.count_nonzero( np.equal( np.argmax(model.predict(x_train),axis = 1 ),y_train ) )/ len(y_train)
accuracy_neg_origin = np.count_nonzero( np.equal( np.argmax(model.predict(x_buggy),axis = 1 ),y_buggy ) )/ len(y_buggy)

accuracy_pos_repair = np.count_nonzero( np.equal( np.argmax(np.dot(f_train,W.transpose()),axis = 1 ),y_train ) )/ len(y_train)
accuracy_neg_repair = np.count_nonzero( np.equal( np.argmax(np.dot(f_buggy,W.transpose()),axis = 1 ),y_buggy ) )/ len(y_buggy)

fout = open(args.log_file,"a+")
fout.write(f"{accuracy_pos_origin},{accuracy_pos_repair},{accuracy_neg_origin},{accuracy_neg_repair}\n")
