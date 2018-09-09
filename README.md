# Overview

Experiment with two approaches to representing sequence of images for action recognition. In the first approach we use Motion History Images 
Toy example for action classification using KTH dataset.

# Data
Dataset downloaded from:
http://www.nada.kth.se/cvap/actions/

All sequences are stored using AVI file format and are available on-line (DIVX-compressed version). Uncompressed version is available on demand. There are 25x6x4=600 video files (however, one is broken) for each combination of 25 subjects, 6 actions and 4 scenarios. 
Available actions are:

1. walking
2. jogging
3. running
4. boxing
5. handwaving
6. handclapping

For details refer to:
"Recognizing Human Actions: A Local SVM Approach",
Christian Schuldt, Ivan Laptev and Barbara Caputo; in Proc. ICPR'04, Cambridge, UK.
# Models
## SVM

## LSTM
### Architecture

### Training setting
LR - 0.0001 \
Optimizer - rmsprop \
Epochs - 100 \
Droput - 0.5

# Results
All results were obtained using cross validation with 5 folds.

| Model                  | Mean accuracy | STD  |
|------------------------|---------------|------|
| SVM on Zernike moments | 0.66          | 0.05 |
| SVM on HOG features    | 0.70          | 0.07 |
| LSTM 10 frames         | 0.80          | 0.04 |
