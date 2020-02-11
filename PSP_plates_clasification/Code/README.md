CNN classification for dentistry project

To use the classifier consider the following parameters:
usage: main.py [-h]
               [--arch {resnet18,resnet50,vgg19_bn,inception_v3,resnetusm}]
               [--val_mode {once,k-fold,randomsampler}]
               [--batch_size BATCH_SIZE] [--workers WORKERS]
               [--k_fold_num K_FOLD_NUM] [--val_num VAL_NUM]
               [--random_seed RANDOM_SEED] [--data_dir DATA_DIR] [--cuda CUDA]
               [--shuffle_dataset SHUFFLE_DATASET] [--pretrained PRETRAINED]
               [--lr LR] [--momentum MOMENTUM] [--weights WEIGHTS]
               [--train TRAIN] [--num_epochs NUM_EPOCHS] [--save SAVE]
               [--image IMAGE] [--folder FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --arch {resnet18,resnet50,vgg19_bn,inception_v3,resnetusm}
                        The network architecture
  --val_mode {once,k-fold,randomsampler}
                        Type of validation you want to use: f.e: k-fold
  --batch_size BATCH_SIZE
                        Input batch size for using the model
  --workers WORKERS     Number of data loading workers
  --k_fold_num K_FOLD_NUM
                        Number of folds you want to use for k-fold validation
  --val_num VAL_NUM     Number of times you want to run the model to get a
                        mean
  --random_seed RANDOM_SEED
                        Random seed to shuffle the dataset
  --data_dir DATA_DIR   Directory were you take images, they have to be
                        separeted by classes
  --cuda CUDA           Use gpu by cuda
  --shuffle_dataset SHUFFLE_DATASET
                        Number of folds you want to use for k-fold validation
  --pretrained PRETRAINED
                        The model will be pretrained with
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum MOMENTUM   momentum
  --weights WEIGHTS     The .pth doc to load as weights
  --train TRAIN         The .pth doc to load as weights
  --num_epochs NUM_EPOCHS
                        number of epochs to train for
  --save SAVE           If you want to save weights and csvs from the trained
                        model or evaluation
  --image IMAGE         The source file path of image you want to process
  --folder FOLDER       The folder where you want to save
