#!/usr/bin/bash
ResPath=MMResRIPPLE
mkdir ${ResPath}
for name in fmnist #cifar10 fmnist #
do
for a in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ##0.6 0.7 0.8 0.9 1.0
do
for b in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.6 0.7 0.8 0.9 1.0
do
python PreForRIPPLE.py   -n ../networks/${name}_adv.h5 \
                    -x ../data/${name}_normal/train_samples.npy \
                    -y ../data/${name}_normal/train_labels.npy \
                    -xa ../data/${name}_adv/train_samples.npy \
                    -ya ../data/${name}_adv/train_labels.npy \
                    -a ${a} \
                    -b ${b} \
                    -d ${ResPath}/${name}_${a}_${b}_samples.npy \
                    -l ${ResPath}/${name}_${a}_${b}_labels.npy
done
done
done
