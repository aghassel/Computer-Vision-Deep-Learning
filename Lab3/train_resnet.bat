@echo off

echo "Activating conda env"
call conda activate ELEC475

echo "Training the model"
python train_resnet.py --batch_size 50 --epochs 110 --lr 0.001 --classes 100 --encoder models/encoder.pth --frontend models/VGGResNet.pth --save_plot graphs/resnet_plot.png
