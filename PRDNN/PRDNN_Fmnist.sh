#!/usr/bin/bash
source ~/repair/Repairenv/bin/activate

#for dir in cifar10
#do
#for layer in 12 14 #6 8
#do
#for size in 2000 #50 100 150 200 500 1000 2000 5000 10000 20000 50000 
#do
network=fmnist
for layer in 9 7
do
for a in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
for b in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do 
for tp in MMRes #HHRes HLResultData LHRes LLRes MMRes
do
python rankingPRDNN.py \
    -n ../networks/${network}_adv.onnx \
    -x ../NewPreprocessing/${tp}/${network}_${a}_${b}_samples.npy \
    -y ../NewPreprocessing/${tp}/${network}_${a}_${b}_labels.npy \
    -l ${layer} \
    -tx ../data/${network}_normal/test_samples.npy \
    -ty ../data/${network}_normal/test_labels.npy \
    -gx ../data/${network}_adv/test_samples.npy \
    -gy ../data/${network}_adv/test_labels.npy \
    -f results/${tp}_${network}_${a}_${b}_${layer}.json
done
done
done
done