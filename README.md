# KTH

Toy example for action classification using KTH dataset.

# Models
## SVM

## LSTM
### Architecture
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input (InputLayer)           (None, 6, 4096)           0         
_________________________________________________________________
LSTM (LSTM)                  (None, 100)               1678800   
_________________________________________________________________
dropout_30 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_30 (Dense)             (None, 6)                 606       
=================================================================
Total params: 1,679,406
Trainable params: 1,679,406
Non-trainable params: 0

### Training setting
LR - 0.0001
Optimizer - rmpsprop
Epochs - 100
Droput - 0.7

# Results
All results were obtained using cross validation with 5 folds.

| Model                  | Mean accuracy | STD  |
|------------------------|---------------|------|
| SVM on Zernike moments | 0.66          | 0.05 |
| SVM on HOG features    | 0.70          | 0.07 |
| LSTM on 10 frames      | 0.72          | 0.02 |
