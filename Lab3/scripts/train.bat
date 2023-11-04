@echo off

cd ..

echo "Activating conda env"
call conda activate ELEC475
echo "Training the model"
python train_custom.py --batch_size 128 --epochs 100 --lr 0.0001 --classes 100 --encoder models/encoder.pth --frontend models/deepResNet5.pth --save_plot graphs/deepResNet4.png

cd scripts
