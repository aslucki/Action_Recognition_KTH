# KTH

Toy example for action classification using KTH dataset.

# Models
## SVM

## LSTM
### Architecture

### Training setting
LR - 0.0001 \
Optimizer - rmsprop \
Epochs - 100 \
Droput - 0.7

# Results
All results were obtained using cross validation with 5 folds.

| Model                  | Mean accuracy | STD  |
|------------------------|---------------|------|
| SVM on Zernike moments | 0.66          | 0.05 |
| SVM on HOG features    | 0.70          | 0.07 |
| LSTM on 10 frames      | 0.72          | 0.02 |
