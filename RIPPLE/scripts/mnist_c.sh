#!/usr/bin
for model in mnist
do
for atk in c
do
##python noSplitPCA.py 
python ../src/SolveWithLP.py -n ../networks/${model}_${atk}.h5 \
		-log ../results/${model}_${atk}_time.txt \
		-r  ../results/${model}_${atk}_patch.txt \
                -b  ../data/${model}_${atk}/fog/train_samples.npy \
		-bl ../data/${model}_${atk}/fog/train_labels.npy \
                -p  ../data/${model}_c/identity/train_samples.npy \
		-pl ../data/${model}_c/identity/train_labels.npy

python ../src/validationLP.py -n ../networks/${model}_${atk}.h5 \
                -log ../results/${model}_${atk}_acc.txt \
                -r  ../results/${model}_${atk}_patch.txt \
                -b  ../data/${model}_${atk}/fog/test_samples.npy \
                -bl ../data/${model}_${atk}/fog/test_labels.npy \
                -p  ../data/${model}_c/identity/test_samples.npy \
                -pl ../data/${model}_c/identity/test_labels.npy
done
done
