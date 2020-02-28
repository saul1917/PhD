
#Binary inbreast test
#Default CIFAR-10 test
#python main.py  --dataset cifar10  --labels data-local/labels/cifar10/1000_balanced_labels/00.txt  --arch cifar_shakeshake26  --consistency 100.0  --consistency-rampup 5  --labeled-batch-size 62  --epochs 180   --lr-rampdown-epochs 200

#test with inbreast
python main.py  --dataset inbreast_binary --arch alexnet_inbreast_binary   --labeled-batch-size 0  --epochs 100   --lr-rampdown-epochs 200 --labels data-local/labels/InBreast/percentage_labeled_0.5/batch_0.txt --eval-subdir train --evaluate False --initial-lr 0.0001 --lr 0.0001 --batch-size 32