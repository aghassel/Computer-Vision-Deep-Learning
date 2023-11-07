@echo off

echo "Training..."

call conda activate ELEC475

python train.py --batch_size 50 --epochs 1000 --lr 0.001 --classes 100 --encoder encoder.pth --frontend resnet.pth --save_plot loss_resnet.pth --model resnet

echo "Done."
