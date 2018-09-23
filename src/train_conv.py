import h5py
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, Input, Dropout, 
                                     Conv2D, BatchNormalization, MaxPooling2D,
                                     Flatten)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow import set_random_seed as tf_random_seed
import argparse

def conv_model(input_shape, n_classes=6):
    """
    Defines model's architecture
    """

    inputs = Input(shape=(input_shape),name='input')
    conv1 = Conv2D(filters=96, kernel_size=(7,7), strides=2)(inputs)
    norm = BatchNormalization()(conv1)
    pool = MaxPooling2D(pool_size=(2,2))(norm)

    conv2 = Conv2D(filters=256, kernel_size=(5,5), strides=2)(pool)
    norm = BatchNormalization()(conv2)

    conv3 = Conv2D(filters=512, kernel_size=(3, 3), strides=1)(norm)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=1)(conv3)

    conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=1)(conv4)
    pool = MaxPooling2D(pool_size=(2, 2))(conv5)

    flattened = Flatten()(pool)
    full6 = Dense(4096)(flattened)
    dropout = Dropout(0.5)(full6)

    full7 = Dense(2048)(dropout)
    dropout = Dropout(0.5)(full7)

    predictions = Dense(n_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def labels_to_one_hot(labels):
    """
    Converts string labels
    to one hot encoding.

    """

    unique_labels = np.unique(labels)
    label_encoding_table = {}

    for ind, name in enumerate(sorted(unique_labels)):
        label_encoding_table[name] = ind

    labels = list(map(lambda x: label_encoding_table[x], labels))
    return to_categorical(labels)


def one_hot_to_label(one_hot, labels):
    unique_labels = np.unique(labels)
    ind = np.argmax(one_hot)

    return sorted(unique_labels)[ind]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True,
            help="Path to a file with resnet data.")
    ap.add_argument("--ds_name", required=True)
    args = vars(ap.parse_args())

    np.random.seed(42)
    tf_random_seed(42)

    data = h5py.File(args['input_file'], 'r')
    labels = labels_to_one_hot(data['labels'][:])
    features = data[args['ds_name']][:]
    data.close()

    scores = []
    kf = KFold(n_splits=5, shuffle=True)
    
    for train_index, test_index in kf.split(labels):
        train_x, train_y = features[train_index], labels[train_index]
        test_x, test_y = features[test_index], labels[test_index]
        
        model = conv_model(input_shape=features[0].shape)
        model.optimizer.lr = 0.0001
        model.fit(train_x, train_y, epochs=100)
        
        scores.append(model.evaluate(test_x, test_y))
    
    scores = np.asarray(scores)
    mean_acc = np.mean(scores[:,1])
    std_acc = np.std(scores[:,1])
    print("Mean accuracy: {}\n STD: {}".format(mean_acc, std_acc))