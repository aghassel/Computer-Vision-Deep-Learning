@echo off 

call conda activate ELEC475

cd ..

echo "Generating Train dataset..."

python Kitti/KittiToYodaROIs.py -i data/Kitti8 -o data/Kitti8_ROIs/train -m train -cuda n -IoU 0.02 -d n 

echo "Generating Test dataset..."

python Kitti/KittiToYodaROIs.py -i data/Kitti8 -o data/Kitti8_ROIs/test -m test -cuda n -IoU 0.02 -d n

cd scripts

echo "Done!"