
"""Experiment to patch an MNIST image-recognition model."""
from collections import defaultdict
import random
from timeit import default_timer as timer
import numpy as np
from pysyrenn import Network
from pysyrenn import ReluLayer,FullyConnectedLayer
from experiments.rankingExperiment import Experiment
from prdnn import ProvableRepair
import argparse
import json
import resource
import gc

soft,hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (128*1024*1024*1024, hard))

class MNISTRepair(Experiment):
    """Attempts to patch networks to be resillient to corruptions."""
    def run(self):
        """Runs the corruption-patching experiment."""
        network = self.load_network()

        if isinstance(network.layers[-1], ReluLayer):
            network = Network(network.layers[:-1])

        for i,Layer in enumerate(network.layers):
            if isinstance(Layer, FullyConnectedLayer):
                print(i)
        self.record_artifact(network, "original", "network")
        self.close()
        self.open()
        #n = self.read_artifact("original")
        self.run_for(network)
    def load_network(self):
        return Network.from_file(self.args.network)
    def run_for(self, network):
        """Runs experiments for a particular # of lines."""
        start = timer()
        experiment = self.experiment
        points = np.load(self.args.trainData)
        labels = np.load(self.args.trainLabels)
        #labels = self.getInput("adv","train",self.atk)
        print("Original ACC:",self.accuracy(network, points, labels))
        #print("Original ACC:",self.accuracy(self.read_artifact("original"), points, labels))
        #exit()
        print(len(labels))
        #Record the SyReNNs and the labels.
        self.record_artifact(
            points, f"{experiment}/train_samples", "pickle")
        self.record_artifact(
            labels, f"{experiment}/train_labels", "pickle")
        
        print("::::", "Layer:", self.args.layer, "::::")
        patcher = ProvableRepair(network, self.args.layer, points, labels)
        patcher.constraint_bufer = 0.001
        patcher.gurobi_crossover = 0
        patcher.gurobi_timelimit = 90 * 60
        patched = patcher.compute()
        syrenn_time = timer() - start
        patcher.timing["syrenn_time"] = syrenn_time
        patcher.timing["total"] += syrenn_time

        self.record_artifact(
            patched, f"{experiment}/patched_{self.args.layer}",
            "pickle" if patched is None else "ddnn")
        self.record_artifact(
            patcher.timing, f"{experiment}/timing_{self.args.layer}", "pickle")

    def analyze(self):
        """Analyze the patched MNIST networks.

        Reports: Time, Drawdown, and Generalization
        """
        experiments = defaultdict(list)
        for artifact in self.artifacts:
            if "timing" not in artifact["key"]:
                continue
            # 10_lines/timing_2
            print(artifact["key"].split("/"))
            eles = artifact["key"].split("/")
            pre = '/'.join(eles[:-1])
            suff = eles[-1]
            experiments[pre].append(suff)

        original_network = self.read_artifact("original")

        stest_lines = np.load(self.args.testData)
        stest_labels = np.load(self.args.testLabels)
        gtest_lines = np.load(self.args.genData)
        gtest_labels = np.load(self.args.genLabels)
        print("Size of drawdown, generalization sets:", len(stest_labels),len(gtest_labels))

        timing_cols = ["layer", "total", "syrenn", "jacobian", "solver",
                       "did_timeout", "drawdown", "generalization"]
        for experiment in experiments.keys():
            print(f"~~~~ Analyzing: {experiment} ~~~~")
            # Get the patched data.
            #train_sample = self.read_artifact(f"{experiment}/train_samples")
            #train_labels = self.read_artifact(f"{experiment}/train_labels")
            train_sample = np.load(self.args.trainData)
            train_labels = np.load(self.args.trainLabels)
            print("Size of repair set:", len(train_labels))

            before = dict({
            "train_corrupted": self.accuracy(original_network, train_sample, train_labels),
            "test_identity": self.accuracy(original_network, stest_lines, stest_labels),
            "test_corrupted": self.accuracy(original_network, gtest_lines, gtest_labels),
            })

            results = self.begin_csv(f"{experiment}/analyzed", timing_cols)
            for layer in [self.args.layer]:
                timing = self.read_artifact(f"{experiment}/timing_{layer}")
                patched = self.read_artifact(f"{experiment}/patched_{layer}")

                record = timing.copy()
                record["layer"] = layer
                record["syrenn"] = record["syrenn_time"]
                del record["syrenn_time"]

                if patched is None:
                    record["drawdown"], record["generalization"] = "", ""
                else:
                    after = dict({
            "train_corrupted": self.accuracy(patched, train_sample, train_labels,1),
            "test_identity": self.accuracy(patched, stest_lines, stest_labels),
            "test_corrupted": self.accuracy(patched, gtest_lines, gtest_labels),
            })
                    print("Layer:", layer)
                    print("\tTime (seconds):", timing["total"])

                    #assert after["train_corrupted"] == 100.

                    record["drawdown"] = (before["test_identity"]
                                          - after["test_identity"])
                    record["generalization"] = (after["test_corrupted"]
                                                - before["test_corrupted"])
                    print("\tDrawdown:", record["drawdown"])
                    print("\tGeneralization:", record["generalization"])
                    print(after)
                    print(before)
                    json.dump({'Time':timing["total"],'before':before,'after':after},open(self.args.outfile,"w"),indent=6)
                self.write_csv(results, record)
        return True

    def compute_accuracies(self, network, train, train_labels, test,
                           test_labels):
        """Compture train, test accuracy for a network."""
        return dict({
            "train_identity": self.accuracy(network, train[0], train_labels),
            "train_corrupted": self.accuracy(network, train[1], train_labels),
            "test_identity": self.accuracy(network, test[0], test_labels),
            "test_corrupted": self.accuracy(network, test[1], test_labels),
        })

    @staticmethod
    def accuracy(network, inputs, labels,mark = 0):
        """Measures network accuracy."""
        net_labels = np.argmax(network.compute(inputs), axis=1)
        if mark== 1:
            pred = network.compute(inputs)
            for i,l in zip(pred,labels):
               if (np.argmax(i) != l):
                  print(i,l)
        return 100. * (np.count_nonzero(np.equal(net_labels, labels))
                       / len(labels))

    @classmethod
    def syrenn_to_points(cls, syrenn, line_labels):
        """Lists all endpoints in an ExactLine/SyReNN representation.

        Returns (points, representatives, labels). Representatives are
        non-vertex points which should have the same activation pattern in the
        network as the corresponding point.
        """
        points, representatives, labels = [], [], []
        for line, label in zip(syrenn, line_labels):
            for start, end in zip(line, line[1:]):
                points.extend([start, end])
                labels.extend([label, label])
                representative = (start + end) / 2.
                representatives.extend([representative, representative])
        return points, representatives, labels

    @staticmethod
    def get_corrupted(split, max_count, only_correct_on=None, corruption="fog"):
        """Returns the desired dataset."""
        random.seed(24)
        np.random.seed(24)

        all_images = [
            np
            .load(f"external/mnist_c/{corruption}/{split}_images.npy")
            .reshape((-1, 28 * 28))
            for corruption in ("identity", corruption)
        ]
        labels = np.load(f"external/mnist_c/identity/{split}_labels.npy")

        indices = list(range(len(labels)))
        random.shuffle(indices)
        labels = labels[indices]
        all_images = [images[indices] / 255. for images in all_images]

        if only_correct_on is not None:
            outputs = only_correct_on.compute(all_images[0])
            outputs = np.argmax(outputs, axis=1)

            correctly_labelled = (outputs == labels)

            all_images = [images[correctly_labelled] for images in all_images]
            labels = labels[correctly_labelled]

        lines = list(zip(*all_images))
        if max_count is not None:
            lines = lines[:max_count]
            labels = labels[:max_count]
        return lines, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--network")
    parser.add_argument("-x","--trainData")
    parser.add_argument("-y","--trainLabels")
    parser.add_argument("-l","--layer",type = int)
    parser.add_argument("-tx","--testData")
    parser.add_argument("-ty","--testLabels")
    parser.add_argument("-gx","--genData")
    parser.add_argument("-gy","--genLabels")   
    parser.add_argument("-f","--outfile") 
    args = parser.parse_args()
    np.random.seed(24)
    random.seed(24)
    MNISTRepair(args).main()

