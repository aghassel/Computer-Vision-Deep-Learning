#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gamma15.out
#SBATCH --error=logs/gamma15.err
#SBATCH --time=5:00:00

CUDA_VISIBLE_DEVICES=1

python3 train.py -content_dir /ingenuity_NAS/dataset/public/coco/images/train2017 -style_dir /ingenuity_NAS/dataset/public/wikiart/images -gamma 15 -e 20 -b 10 -l encoder.pth -s decoder_g15_10k.pth -p decoder_g15_10k.png -cuda Y