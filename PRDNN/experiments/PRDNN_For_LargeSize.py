"""Repair SqueezeNet on the NAE dataset.
"""
from ast import Add
from collections import defaultdict
import random
import numpy as np
# pylint: disable=import-error
from pysyrenn import Network
from pysyrenn import FullyConnectedLayer, Conv2DLayer
from prdnn import ProvableRepair
from expForLargeSize import Experiment
import pandas as pd
import argparse
import resource


def limit_memory(maxsize):
        soft,hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,(maxsize,hard))

limit_memory(1024*1024*1024*64)


class SqueezenetRepair(Experiment):
    """Repairs Imagenet with the NAE dataset (Hendrycks et al.)"""
    def run(self,networkPath,examples,labels,normalize):
        print(networkPath) #"""Repair Squeezenet model and record patched versions."""
        network = Network.from_file(networkPath)
        # Get the trainset and record it.
        train_inputs, train_labels = getDataAndLabel(examples,labels,normalize)

        self.record_artifact(train_inputs, f"train_inputs", "pickle")
        self.record_artifact(train_labels, f"train_labels", "pickle")

        # Record the network before patching.
        self.record_artifact(network, f"pre_patching", "network")

        # All the layers we can patch.
        patchable = [i for i, layer in enumerate(network.layers)
                     if isinstance(layer, (FullyConnectedLayer, Conv2DLayer))]
        #n_rows = int(input("How many rows of Table 1 to generate (1, 2, 3, or 4): "))
        for n_points in [100000]:
            print("~~~~", "Points:", n_points, "~~~~")
            for layer in patchable[-1:]:
                print("::::", "Layer:", layer, "::::")
                key = f"{n_points}_{layer}"

                patcher = ProvableRepair(
                    network, layer,
                    train_inputs[:n_points], train_labels[:n_points])
                patcher.batch_size = 8
                patcher.gurobi_timelimit = (n_points // 10) * 60
                patcher.gurobi_crossover = 0

                patched = patcher.compute()

                self.record_artifact(patcher.timing, f"{key}/timing", "pickle")
                self.record_artifact(
                    patched, f"{key}/patched",
                    "ddnn" if patched is not None else "pickle")

    def analyze(self,examples,labels,normalize):
        """Compute drawdown statistics for patched models."""
        print("~~~~ Results ~~~~")
        # Get the datasets and compute pre-patching accuracy.
        network = self.read_artifact("pre_patching")
        train_inputs = self.read_artifact("train_inputs")
        train_labels = self.read_artifact("train_labels")

        test_inputs, test_labels = getDataAndLabel(examples,labels,normalize)

        original_train_accuracy = self.accuracy(
            network, train_inputs, train_labels)
        original_test_accuracy = self.accuracy(
            network, test_inputs, test_labels)
        print("Max size of repair set:", len(train_inputs))
        print("Size of drawdown set:", len(test_inputs))
        print("Buggy network repair set accuracy:", original_train_accuracy)
        print("Buggy network drawdown set accuracy:", original_test_accuracy)

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
                    new_train_accuracy = self.accuracy(patched,
                                                       train_inputs[:n_points],
                                                       train_labels[:n_points])
                    new_test_accuracy = self.accuracy(patched,
                                                      test_inputs,
                                                      test_labels)
                    record["efficacy"] = new_train_accuracy
                    record["drawdown"] = (original_test_accuracy
                                          - new_test_accuracy)
                    records_for_row.append(record)
                else:
                    record["efficacy"] = 0
                    record["drawdown"] = 0

                record["layer"] = layer
                self.write_csv(n_points_csvs[n_points], record)
                del record["layer"]
                record["points"] = n_points
                self.write_csv(layer_csvs[layer], record)
            best_record = min(records_for_row, key=lambda record: record["drawdown"])
            print("\tBest drawdown:", best_record["drawdown"])
            print("\tTotal time for best drawdown (seconds):", best_record["total"])
        return True

    def main(self,args):
        """Main experiment harness.
        """
        self.run(args.network,args.buggy_examples,args.buggy_labels,args.normalizeb)
        self.close()
        self.open()
        did_modify = self.analyze(args.test_generality_examples,args.test_generality_labels,args.normalizeg)
        did_modify = self.analyze(args.test_origin_examples,args.test_origin_labels,args.normalizeo)
        self.close(tar=did_modify)

    @staticmethod
    def accuracy(network, inputs, labels):
        """Computes accuracy on a test set."""
        out = np.argmax(network.compute(inputs), axis=1)
        return 100. * np.count_nonzero(np.equal(out, labels)) / len(labels)

def Add_Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--network",help="The network to repair")
    parser.add_argument("-b","--buggy_examples",help = "The buggy examples")
    parser.add_argument("-bl","--buggy_labels",help = "The buggy examples")
    parser.add_argument("-normb","--normalizeb",type=bool,default = False)
    parser.add_argument("-to","--test_origin_examples",help = "The test examples")
    parser.add_argument("-tol","--test_origin_labels",help = "The test examples")
    parser.add_argument("-normo","--normalizeo",type=bool,default = False)
    parser.add_argument("-tg","--test_generality_examples",help = "The test examples")
    parser.add_argument("-tgl","--test_generality_labels",help = "The test examples") 
    parser.add_argument("-normg","--normalizeg",type=bool,default = False)  
    args = parser.parse_args()
    return args

def getDataAndLabel(examples,labels,normalize):
    normb = 255.0 if normalize else 1.0
    data = pd.read_csv(examples,header=None)
    result = []
    for i in range(len(data)):
                sample = data.loc[i].values #.reshape(shape)
                result.append(sample)
    samples = np.array(result,dtype=np.float32)/normb
    labels = pd.read_csv(labels,header=None)
    n = []
    for i in range(len(labels)):
                n.append(labels.loc[i].iloc[0])
    labels = np.array(n)
    return samples,labels

if __name__ == "__main__":
    np.random.seed(24)
    random.seed(24)
    args = Add_Parser()
    SqueezenetRepair(args.network).main(args)
