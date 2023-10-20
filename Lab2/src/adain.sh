#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gamma2.50.out
#SBATCH --error=logs/gamma2.50.err
#SBATCH --time=6:00:00

CUDA_VISIBLE_DEVICES=2

python3 train.py -content_dir /ingenuity_NAS/dataset/public/coco/images/train2017/ -style_dir /ingenuity_NAS/dataset/public/wikiart/ -gamma 2.5 -e 20 -b 50 -l encoder.pth -s decoder_g250_10k.pth -p decoder_g250_10k.png -cuda Y