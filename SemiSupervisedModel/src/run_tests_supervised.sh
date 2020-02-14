#!/bin/bash
echo 'Executing Breast Image classification tests for fully supervised Alexnet'
conda activate saul
echo 'Tests with complete dataset'
python SemiSupervisedModelController.py --splits_unlabeled 0
echo 'Tests with 50% of the labeled dataset'
echo 'Tests with 50% of the labeled dataset: Fold 1'
python SemiSupervisedModelController.py --splits_unlabeled 2 --current_fold 1
python SemiSupervisedModelController.py --splits_unlabeled 2 --current_fold 2

