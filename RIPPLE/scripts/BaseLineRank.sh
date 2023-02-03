#!/usr/bin
for model in mnist fmnist cifar10 #mnist cifar10 fmnist #cifar10
do
for a in  1.0 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 
do
for b in  1.0 # 0.1 0.2 0.3 0.5 1.0
do
##python noSplitPCA.py 
python ../src/SolveWithLP.py -n ../../networks/${model}_adv.h5 \
		-log ../results/MM${model}_${a}_${b}_time.txt \
		-r  ../results/MM${model}_${a}_${b}_patch.txt \
                -b  ../../NewPreprocessing/MMResRIPPLE/${model}_${a}_${b}_samples.npy \
		-bl ../../NewPreprocessing/MMResRIPPLE/${model}_${a}_${b}_labels.npy \
                -p  ../../NewPreprocessing/MMResRIPPLE/${model}_${a}_${b}_samples.npy \
		-pl ../../NewPreprocessing/MMResRIPPLE/${model}_${a}_${b}_labels.npy  >> ../results/MM${model}_${a}_${b}_log.txt

python ../src/validationLP.py -n ../../networks/${model}_adv.h5 \
                -log ../results/MM${model}_${a}_${b}_time.txt \
		-r  ../results/MM${model}_${a}_${b}_patch.txt \
                -b  ../../data/${model}_adv/test_samples.npy \
                -bl ../../data/${model}_adv/test_labels.npy \
                -p  ../../data/${model}_normal/test_samples.npy \
                -pl ../../data/${model}_normal/test_labels.npy
done
done
done
