#!/usr/bin/bash
ResPath=MMRes
mkdir ${ResPath}
for name in mnist cifar10 fmnist
do
for a in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ##0.6 0.7 0.8 0.9 1.0
do
for b in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.6 0.7 0.8 0.9 1.0
do
python Top1010.py   -n ../networks/${name}_adv.h5 \
                    -x ../data/${name}_adv/train_samples.npy \
                    -y ../data/${name}_adv/train_labels.npy \
                    -s 1000 \
                    -a ${a} \
                    -b ${b} \
                    -d ${ResPath}/${name}_${a}_${b}_samples.npy \
                    -l ${ResPath}/${name}_${a}_${b}_labels.npy
done
done
done
