#!/usr/bin/bash
for type in poisoned
do
for exp in  cifar10
do 
python PRDNN_For_LargeSize.py -n ../../formal/networks/${exp}_${type}.onnx \
                  -b ../../formal/data/${exp}_${type}/train_samples.csv \
                  -bl ../../formal/data/${exp}_${type}/train_labels.csv \
                  -to ../../formal/data/${exp}_normal/validation_samples.csv  \
                  -tol ../../formal/data/${exp}_normal/validation_labels.csv \
                  -tg ../../formal/data/${exp}_${type}/validation_samples.csv \
                  -tgl ../../formal/data/${exp}_${type}/validation_labels.csv \
                   >> templog.txt
done
done
