@echo off

echo "Training..."

call conda activate ELEC475

python train.py --batch_size 50 --epochs 100 --lr 0.001 --classes 100 --encoder encoder.pth --frontend resnet2.pth --save_plot loss_resnet2.pth --model resnet

echo "Done."
