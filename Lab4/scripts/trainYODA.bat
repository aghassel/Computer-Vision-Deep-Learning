@echo off

cd ..

call conda activate ELEC475

python train_YODA_classifier.py --batch_size 75 --epochs 25 --lr 0.001 --model_name mod --momentum 0.9 --data_dir data/Kitti8_ROIs/ --save_dir models/ --cuda True --loss_plot_dir graphs/

cd scripts