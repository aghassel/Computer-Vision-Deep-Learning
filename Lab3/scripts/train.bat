@echo off

cd ..

echo "Activating conda env"
call conda activate ELEC475

echo "Training the model"
python train_custom.py --batch_size 75 --epochs 100 --lr 0.001 --classes 100 --encoder models/encoder.pth --frontend models/deepResNet4.pth --save_plot graphs/deepResNet4.png --resume_training True

cd scripts
