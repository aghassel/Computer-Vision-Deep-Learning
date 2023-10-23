#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gamma10.0_10k.out
#SBATCH --error=logs/gamma10.0_10k.err
#SBATCH --time=6:00:00

CUDA_VISIBLE_DEVICES=3

python3 train.py -content_dir /ingenuity_NAS/dataset/public/coco/images/train2017/ -style_dir /ingenuity_NAS/dataset/public/wikiart/ -gamma 10.0 -e 20 -b 70 -l models/encoder.pth -s decoder_g10.0_10k.pth -p decoder_g10.0_10k.png -cuda Y