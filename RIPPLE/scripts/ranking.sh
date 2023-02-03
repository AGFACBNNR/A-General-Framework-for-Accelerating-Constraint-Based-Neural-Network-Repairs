#!/usr/bin
for model in normmnist #cifar10 #normmnist #cifar10
do
for atk in adv
do
for method in pd #pv pe
do
for threshold in 0.1 #1.0 #0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #80 85 90 95 
do
##python noSplitPCA.py 
python ../src/RankingS.py -n ../networks/${model}_${atk}.h5 \
		-log ../results/ranking/${model}_${atk}_${method}_${threshold}_time.txt \
		-r  ../results/ranking/${model}_${atk}_${method}_${threshold}_patch.txt \
                -b  ../data/${model}_${atk}/train_samples.npy \
		-bl ../data/${model}_${atk}/train_labels.npy \
                -p  ../data/${model}_normal/train_samples.npy \
		-pl ../data/${model}_normal/train_labels.npy \
                -m  ${method} \
                -t  ${threshold} \
		 #> ../results/ranking/${model}_${atk}_${method}_${threshold}_log.txt

python ../src/validationLP.py -n ../networks/${model}_${atk}.h5 \
                -log ../results/ranking/${model}_${atk}_${method}_${threshold}_acc.txt \
                -r  ../results/ranking/${model}_${atk}_${method}_${threshold}_patch.txt \
                -b  ../data/${model}_${atk}/test_samples.npy \
                -bl ../data/${model}_${atk}/test_labels.npy \
                -p  ../data/${model}_normal/test_samples.npy \
                -pl ../data/${model}_normal/test_labels.npy 
done
done
done
done

#for model in mnist cifar10
#do
#for atk in poisoned
#do
#for method in pd #pv pe
#do
#for threshold in 75 #80 85 90 95 
#do
##python noSplitPCA.py 
#python ../src/RankingS.py -n ../networks/${model}_${atk}.h5 \
#                -log ../results/ranking/${model}_${atk}_${method}_${threshold}_time.txt \
#                -r  ../results/ranking/${model}_${atk}_${method}_${threshold}_patch.txt \
#                -b  ../data/${model}_${atk}/train_samples.npy \
#                -bl ../data/${model}_${atk}/train_labels.npy \
#                -p  ../data/${model}_normal/train_samples.npy \
#                -pl ../data/${model}_normal/train_labels.npy \
#                -m  ${method} \
#                -t  ${threshold} \
#                 > ../results/ranking/${model}_${atk}_${method}_${threshold}_log.txt
#
#python ../src/validationLP.py -n ../networks/${model}_${atk}.h5 \
#                -log ../results/ranking/${model}_${atk}_${method}_${threshold}_acc.txt \
#                -r  ../results/ranking/${model}_${atk}_${method}_${threshold}_patch.txt \
#                -b  ../data/${model}_${atk}/test_samples.npy \
#                -bl ../data/${model}_${atk}/test_labels.npy \
#                -p  ../data/${model}_normal/test_samples.npy \
#                -pl ../data/${model}_normal/test_labels.npy 
#done
#done
#done
#done
