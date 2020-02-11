Pytorch - Dentistry project


You can easily test the output masks on your images via the CLI.

To see all options: python predict.py -h

To predict a single image and save it:

python predict.py -i image.jpg -o output.jpg

To predict a multiple images and show them without saving them:

python predict.py -i image1.jpg image2.jpg --viz --no-save

You can use the cpu-only version with --cpu.

You can specify which model file to use with --model MODEL.pth.

Training
python train.py -h should get you started. A proper CLI is yet to be added.

Taken from: https://github.com/milesial/Pytorch-UNet

Have in mind to use images that their width and height are multiples of 32. 
-Ari 
