"""Experiment to patch an MNIST image-recognition model."""
from collections import defaultdict
import random
from timeit import default_timer as timer
import numpy as np
from pysyrenn import Network
from pysyrenn import ReluLayer,FullyConnectedLayer
from experiments.experiment import Experiment
from prdnn import ProvableRepair
import argparse

def accuracy(network, inputs, labels):
        """Measures network accuracy."""
        net_labels = np.argmax(network.compute(inputs), axis=1)
        return 100. * (np.count_nonzero(np.equal(net_labels, labels))
                       / len(labels))

dir = 'cifar10'
n = Network.from_file(f'../networks/{dir}_adv.onnx')
train_x = np.load(f'../NewPreprocessing/MMRes/{dir}_1.0_1.0_samples.npy')
train_y = np.load(f'../NewPreprocessing/MMRes/{dir}_1.0_1.0_labels.npy')
print(n.compute(train_x))
print(train_y.shape)
print(accuracy(n,train_x,train_y))
#train_x = np.load(f'../data/{dir}_adv/test_samples.npy')
#train_y = np.load(f'../data/{dir}_adv/test_labels.npy')
#print(accuracy(n,train_x,train_y))
#train_x = np.load(f'../data/{dir}_normal/test_samples.npy')
#train_y = np.load(f'../data/{dir}_normal/test_labels.npy')
#print(accuracy(n,train_x,train_y))
#train_x = np.load(f'../data/{dir}_normal/train_samples.npy')
#train_y = np.load(f'../data/{dir}_normal/train_labels.npy')
#print(accuracy(n,train_x,train_y))
