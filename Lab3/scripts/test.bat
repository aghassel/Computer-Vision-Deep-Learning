@echo off
cd ..
echo "Activating conda env"
call conda activate ELEC475

echo "Testing the model"
python test.py --batch_size 4 --classes 100 --encoder models/encoder.pth --frontend models/VGGResNet.pth 

cd scripts
```