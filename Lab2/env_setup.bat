@echo off

set ENV_NAME=ELEC475

conda activate %ENV_NAME% || (
    conda create -n %ENV_NAME% --file env.yml
    conda activate %ENV_NAME%
)
