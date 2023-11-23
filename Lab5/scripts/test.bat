@echo off

cd ..

call conda activate ELEC475

python test.py --batch_size 75 --epochs 50 --lr 0.001 --model_name mod --momentum 0.9 --data_dir data/ --save_dir models/ --cuda True --loss_plot_dir graphs/ --threads 12

cd scripts