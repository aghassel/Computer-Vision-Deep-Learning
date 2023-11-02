@echo off

echo "Activating conda env"
call conda activate ELEC475

echo "Training the model"
python train.py --batch_size 500 --epochs 50 --lr 0.001 --classes 100 --encoder encoder.pth --frontend vit.pth --save_plot vit_plot.png
