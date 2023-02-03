#!/bin/sh
#python compare.py ../../formal/networks/mnist_poisoned mnist_poisoned validation mnist_normal
#python compare.py ../../formal/networks/normmnist_adv normmnist_adv train normmnist_adv
#python compare.py ../../formal/networks/cifar10_adv cifar10_adv train cifar10_normal
#python compare.py normmnist_adv train True normmnist_normal #>> mndist_normal_Result.txt
python compare.py cifar10_normal train True cifar10_normal 1
#1 3 6 8 12 14
