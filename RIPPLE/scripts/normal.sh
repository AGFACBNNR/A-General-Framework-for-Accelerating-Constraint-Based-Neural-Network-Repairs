#!/usr/bin
for model in cifar10 mnist
do
for atk in normal
do
##python noSplitPCA.py 
python ../src/SolveWithLP.py -n ../networks/${model}_${atk}.h5 \
		-log ../results/${model}_${atk}_time.txt \
		-r  ../results/${model}_${atk}_patch.txt \
                -b  ../data/${model}_${atk}/train_samples.npy \
		-bl ../data/${model}_${atk}/train_labels.npy \
                -p  ../data/${model}_normal/train_samples.npy \
		-pl ../data/${model}_normal/train_labels.npy

python ../src/validationLP.py -n ../networks/${model}_${atk}.h5 \
                -log ../results/${model}_${atk}_acc.txt \
                -r  ../results/${model}_${atk}_patch.txt \
                -b  ../data/${model}_${atk}/test_samples.npy \
                -bl ../data/${model}_${atk}/test_labels.npy \
                -p  ../data/${model}_normal/test_samples.npy \
                -pl ../data/${model}_normal/test_labels.npy
done
done

