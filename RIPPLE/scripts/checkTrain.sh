#!/usr/bin
for model in normmnist cifar10
do
for atk in adv
do
python ../src/validationLP.py -n ../networks/${model}_${atk}.h5 \
                -log ../results/${model}_${atk}_Tacc.txt \
                -r  ../results/${model}_${atk}_patch.txt \
                -b  ../data/${model}_${atk}/train_samples.npy \
                -bl ../data/${model}_${atk}/train_labels.npy \
                -p  ../data/${model}_${atk}/test_samples.npy \
                -pl ../data/${model}_${atk}/test_labels.npy
done
done

for model in mnist cifar10
do
for atk in normal
do
python ../src/validationLP.py -n ../networks/${model}_${atk}.h5 \
                -log ../results/${model}_${atk}_Tacc.txt \
                -r  ../results/${model}_${atk}_patch.txt \
                -b  ../data/${model}_${atk}/train_samples.npy \
                -bl ../data/${model}_${atk}/train_labels.npy \
                -p  ../data/${model}_normal/test_samples.npy \
                -pl ../data/${model}_normal/test_labels.npy
done
done
