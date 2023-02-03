"""Repair SqueezeNet on the NAE dataset.
"""
from collections import defaultdict
import random
import numpy as np
# pylint: disable=import-error
from pysyrenn import Network
from pysyrenn import ReluLayer, NormalizeLayer
from pysyrenn import FullyConnectedLayer, Conv2DLayer
from prdnn import ProvableRepair
from experiment import Experiment
from keras.models import load_model
from sys import argv
import pandas as pd
import resource
import psutil


def limit_memory(maxsize):
	soft,hard = resource.getrlimit(resource.RLIMIT_AS)
	resource.setrlimit(resource.RLIMIT_AS,(maxsize,hard))

limit_memory(1024*1024*1024*64)

#argv1:networkName argv2: TrainPart argv3: OnlyBug argv4: TestName
networkDir = "../../formal/networks/"
model = load_model(f"{networkDir}{argv[1]}.h5")
shape = model.layers[0].input.shape[1:]
targetLayer = int(argv[5])
def load_data(shape,dirName,part='train',bug=False):
	if (bug):
		data = pd.read_csv(f"../../formal/Bugs/{dirName}_{part}_samples_bugs.csv",header=None)
	else:
		data = pd.read_csv(f"../../formal/data/{dirName}/{part}_samples.csv",header=None)
	result = []
	for i in range(len(data)):
		sample = data.loc[i].values.reshape(shape)
		result.append(sample)
	samples = np.array(result,dtype=np.float32)
	
	if (bug):
		labels = pd.read_csv(f"../../formal/Bugs/{dirName}_{part}_labels_bugs.csv",header=None)
	else:
		labels = pd.read_csv(f"../../formal/data/{dirName}/{part}_labels.csv",header=None)
	n = []
	for i in range(len(labels)):
		n.append(labels.loc[i].iloc[0])
	labels = np.array(n)
	return samples,labels

class SqueezenetRepair(Experiment):
    """Repairs Imagenet with the NAE dataset (Hendrycks et al.)"""
    def run(self):
        """Repair Squeezenet model and record patched versions."""
        network = self.load_network(f"{networkDir}{argv[1]}.onnx")
        # Get the trainset and record it.
        train_inputs,train_labels = load_data(shape,argv[1],argv[2],argv[3]=="True")
        
        self.record_artifact(train_inputs, f"train_inputs", "pickle")
        self.record_artifact(train_labels, f"train_labels", "pickle")

        # Record the network before patching.
        self.record_artifact(network, f"pre_patching", "network")

        # All the layers we can patch.
        patchable = [i for i, layer in enumerate(network.layers)
                     if isinstance(layer, (FullyConnectedLayer, Conv2DLayer))]
        n_rows = 1
        for n_points in [800][:n_rows]:
            print("~~~~", "Points:", n_points, "~~~~")
            for layer in patchable:
                if (layer != targetLayer):
                  continue
                print("::::", "Layer:", layer, "::::")
                key = f"{n_points}_{layer}"

                patcher = ProvableRepair(
                    network, layer,
                    train_inputs[:n_points], train_labels[:n_points])
                patcher.batch_size = 8
                patcher.gurobi_timelimit = (n_points // 10) * 60
                patcher.gurobi_crossover = 0

                patched = patcher.compute()
                
                print(patcher.timing)
                self.record_artifact(patcher.timing, f"{key}/timing", "pickle")
                self.record_artifact(
                    patched, f"{key}/patched",
                    "ddnn" if patched is not None else "pickle")

    def analyze(self):
        """Compute drawdown statistics for patched models."""
        print("~~~~ Results ~~~~")
        # Get the datasets and compute pre-patching accuracy.
        network = self.read_artifact("pre_patching")
        train_inputs = self.read_artifact("train_inputs")
        train_labels = self.read_artifact("train_labels")
        
        test_inputs,test_labels = load_data(shape,argv[4],"validation")
        gen_inputs,gen_labels = load_data(shape,argv[1],"validation")
        
        original_train_accuracy = self.accuracy(
            network, train_inputs, train_labels)
        original_test_accuracy = self.accuracy(
            network, test_inputs, test_labels)
        original_gen_accuracy = self.accuracy(
            network, gen_inputs, gen_labels)
        print("Max size of repair set:", len(train_inputs))
        print("Size of drawdown set:", len(test_inputs))
        print("Size of generalization set:", len(gen_inputs))
        
        print("Buggy network repair set accuracy:", original_train_accuracy)
        print("Buggy network drawdown set accuracy:", original_test_accuracy)
        print("Buggy network generalization set accuracy:", original_gen_accuracy)
        
        # Get info about the patch runs.
        by_n_points = defaultdict(list)
        by_layer = defaultdict(list)
        for artifact in self.artifacts:
            artifact = artifact["key"]
            if "timing" not in artifact:
                continue
            key = artifact.split("/")[0]
            n_points, layer = map(int, key.split("_"))
            by_n_points[n_points].append(layer)
            by_layer[layer].append(n_points)

        timing_cols = ["total", "jacobian", "solver", "did_timeout",
                       "efficacy", "drawdown"]
        n_points_csvs = dict({
            n_points:
                self.begin_csv(f"{n_points}_points", ["layer"] + timing_cols)
            for n_points in by_n_points.keys()
        })
        layer_csvs = dict({
            layer: self.begin_csv(f"{layer}_layer", ["points"] + timing_cols)
            for layer in by_layer.keys()
        })
        for n_points in sorted(by_n_points.keys()):
            print("~~~~~", "Points:", min(int(n_points), len(train_inputs)), "~~~~~")
            records_for_row = []
            for layer in sorted(by_n_points[n_points]):
                timing = self.read_artifact(f"{n_points}_{layer}/timing")
                record = timing.copy()

                patched = self.read_artifact(f"{n_points}_{layer}/patched")
                if patched is not None:
                    print(f"layer : {layer}")
                    new_train_accuracy = self.accuracy(patched,
                                                       train_inputs[:n_points],
                                                       train_labels[:n_points])
                    new_test_accuracy = self.accuracy(patched,
                                                      test_inputs,
                                                      test_labels)
                    new_gen_accuracy = self.accuracy(patched,
                                                      gen_inputs,
                                                      gen_labels)
                    print(f"efficacy : {new_train_accuracy}")
                    print(f"drawdown : {original_test_accuracy-new_test_accuracy}")
                    print(f"genAcc : {new_gen_accuracy}")
                    print(f"TotalAcc : {(new_test_accuracy*len(test_labels)+new_gen_accuracy*len(gen_labels))/(len(test_labels)+len(gen_labels))}")
                    print(f"Time : {timing}")
        return True

    @staticmethod
    def accuracy(network, inputs, labels):
        """Computes accuracy on a test set."""
        out = np.argmax(network.compute(inputs), axis=1)
        return 100. * np.count_nonzero(np.equal(out, labels)) / len(labels)

if __name__ == "__main__":
    np.random.seed(24)
    random.seed(24)
    SqueezenetRepair("mnist0_adv").main()
