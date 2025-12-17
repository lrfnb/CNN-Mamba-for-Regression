## CMCNet: Enhancing Face Image Super-Resolution through CNN-Mamba Collaboration

## Installation and Requirements 
I have trained and tested the codes on
- Ubuntu 20.04
- CUDA 11.8  
- Python 3.8

### Pretrained models and test results
The **pretrained models** and **test results** are being reorganized.

### Train the Model

```
# Train Code
python train.py --gpus 1 --name ResolutionRegression --model resolutionregressionnet --Gnorm "bn" --lr 0.0001 --beta1 0.9 --load_size 128  --dataroot /data2/lrf/data/spatial-resolution_0.1m/train_mix --dataset_name remotesensingresolution --batch_size 12 --total_epochs 100 --visual_freq 20 --print_freq 50 --save_latest_freq 1000
```


### Test with Pretrained Models
```
python test_regression.py
```


