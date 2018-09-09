import h5py
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow import set_random_seed as tf_random_seed
import argparse

def lstm_model(input_shape, n_classes=6):
    """
    Defines model's architecture
    """

    inputs = Input(shape=(input_shape),name='input')
    lstm_out = LSTM(100, name='LSTM')(inputs)
    dropout = Dropout(0.5)(lstm_out)
    predictions = Dense(n_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
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
    args = vars(ap.parse_args())

    np.random.seed(42)
    tf_random_seed(42)

    data = h5py.File(args['input_file'], 'r')
    labels = labels_to_one_hot(data['labels'][:])
    features = data['resnet50'][:]
    data.close()

    scores = []
    kf = KFold(n_splits=5, shuffle=True)
    
    for train_index, test_index in kf.split(labels):
        train_x, train_y = features[train_index], labels[train_index]
        test_x, test_y = features[test_index], labels[test_index]
        
        model = lstm_model(input_shape=(10,2048))
        model.optimizer.lr = 0.0001
        model.fit(train_x, train_y, epochs=100)
        
        scores.append(model.evaluate(test_x, test_y))
    
    scores = np.asarray(scores)
    mean_acc = np.mean(scores[:,1])
    std_acc = np.std(scores[:,1])
    print("Mean accuracy: {}\n STD: {}".format(mean_acc, std_acc))
