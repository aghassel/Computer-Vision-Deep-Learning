#!/bin/bash 
# bash file to run train code

python train.py --batch_size 128 --epochs 50 --lr 0.001 --classes 100 --encoder encoder.pth --frontend frontend.pth