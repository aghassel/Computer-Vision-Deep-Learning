@echo off
cd ..
call conda activate ELEC475
python train_YODA_classifier.py
cd scripts