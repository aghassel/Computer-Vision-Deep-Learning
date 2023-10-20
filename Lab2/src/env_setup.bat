@echo off

set ENV_NAME=ELEC475
conda activate %ENV_NAME% || conda env create -f env.yml && conda activate %ENV_NAME%

