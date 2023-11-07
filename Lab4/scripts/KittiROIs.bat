@echo off 

call conda activate ELEC475

cd ..

python Kitti/KittiToYodaROIs.py -i data/Kitti8 -o data/Kitti8_ROIs/train -m train -cuda y -IoU 0.02 -d y 

python Kitti/KittiToYodaROIs.py -i data/Kitti8 -o data/Kitti8_ROIs/test -m test -cuda y -IoU 0.02 -d y