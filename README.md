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
Scikit-learn implementation of LinearSVC.
Used cross-validation with 5 folds to find optimal *C* parameter from:
>[1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

## LSTM
### Architecture

    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           (None, 10, 2048)          0         
    _________________________________________________________________
    LSTM (LSTM)                  (None, 100)               859600    
    _________________________________________________________________
    dropout (Dropout)            (None, 100)               0         
    _________________________________________________________________
    dense (Dense)                (None, 6)                 606       
    =================================================================
    Total params: 860,206
    Trainable params: 860,206
    Non-trainable params: 0


### Training setting
LR - 0.0001 \
Optimizer - rmsprop \
Max Epochs - 100 \
Droput - 0.5

Trained with an early stopping callback (training stops when accuracy haven't improved in *3* consecutive epochs).

## Convolutional network
### Architecture
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           (None, 120, 160, 18)      0         
    _________________________________________________________________
    conv2d_58 (Conv2D)           (None, 57, 77, 96)        84768     
    _________________________________________________________________
    batch_normalization_24 (Batc (None, 57, 77, 96)        384       
    _________________________________________________________________
    max_pooling2d_22 (MaxPooling (None, 28, 38, 96)        0         
    _________________________________________________________________
    conv2d_59 (Conv2D)           (None, 12, 17, 256)       614656    
    _________________________________________________________________
    batch_normalization_25 (Batc (None, 12, 17, 256)       1024      
    _________________________________________________________________
    conv2d_60 (Conv2D)           (None, 10, 15, 512)       1180160   
    _________________________________________________________________
    conv2d_61 (Conv2D)           (None, 8, 13, 512)        2359808   
    _________________________________________________________________
    conv2d_62 (Conv2D)           (None, 6, 11, 512)        2359808   
    _________________________________________________________________
    max_pooling2d_23 (MaxPooling (None, 3, 5, 512)         0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 7680)              0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 4096)              31461376  
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 4096)              0         
    _________________________________________________________________
    dense_16 (Dense)             (None, 2048)              8390656   
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 2048)              0         
    _________________________________________________________________
    dense_17 (Dense)             (None, 6)                 12294     
    =================================================================
    Total params: 46,464,934
    Trainable params: 46,464,230
    Non-trainable params: 704
    
### Training setting
LR - 0.0001 \
Optimizer - rmsprop \
Max Epochs - 100 \
Droput - 0.5

Trained with an early stopping callback (training stops when accuracy haven't improved in *3* consecutive epochs).

# Results
All results were obtained using cross validation with 5 folds.

| Model                  | Mean accuracy | STD  |
|------------------------|---------------|------|
| Autoencoder features   | 0.656         | 0.062|
| SVM on Zernike moments | 0.656         | 0.056|
| SVM on HOG features    | 0.684         | 0.060|
| LSTM on 10 frames      | 0.759         | 0.018|
| LSTM on 10 shuffled frames | 0.664         | 0.038|
| CONV on optical flows [Zisserman]      | 0.527         |0.05|
|Linear model on resnet features [Zisserman]    | 0.721         |0.021|
|SVM on output from CONV on optical flows and linear model on resnet  [Zisserman]   | 0.721         |0.022|

[Zisserman]: https://arxiv.org/pdf/1406.2199.pdf
