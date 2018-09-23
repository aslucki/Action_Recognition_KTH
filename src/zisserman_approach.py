import h5py
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Dense, Input, Dropout, 
                                     Conv2D, BatchNormalization, MaxPooling2D,
                                     Flatten)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
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

def linear_model(input_shape, n_classes=6):
    inputs = Input(shape=(input_shape),name='input')

    full7 = Dense(2048)(inputs)
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

def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return a


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet_features_file", required=True,
            help="Path to a file with resnet data.")
    ap.add_argument("--optical_flow_features_file", required=True,
            help="Path to a file with resnet data.")
    args = vars(ap.parse_args())

    np.random.seed(42)
    tf_random_seed(42)

    resnet_file = h5py.File(args['resnet_features_file'], 'r')
    labels = resnet_file['labels'][:]
    labels_one_hot = labels_to_one_hot(labels)
    resnet_features = resnet_file['resnet50'][:]

    # Randomly select single frame
    resnet_features = disarrange(resnet_features, axis=-2)
    resnet_features = resnet_features[:,0,...]
    resnet_file.close()

    optical_flow_file = h5py.File(args['optical_flow_features_file'], 'r')
    optical_flow_features = optical_flow_file['optical_flow'][:]
    optical_flow_file.close()


    svm_parameters = [10**x for x in range(-5,5)]

    
    resnet_scores = []
    optical_flow_scores = []
    svm_scores = []


    early_stopper = EarlyStopping(monitor='acc', patience=3)
    kf = KFold(n_splits=5, shuffle=True)
    split = 0
    for train_index, test_index in kf.split(labels):
        print(split)
        train_x_resnet = resnet_features[train_index]
        train_x_optical_flow = optical_flow_features[train_index]
        train_y = labels_one_hot[train_index]
        train_y_svm = labels[train_index]
        
        test_x_resnet = resnet_features[test_index]
        test_x_optical_flow = optical_flow_features[test_index]
        test_y = labels_one_hot[test_index]
        test_y_svm = labels[test_index]
        
        
        # Train model on resnet features
        print("Training model on resnet features")
        model_resnet = linear_model(input_shape=resnet_features[0].shape)
        model_resnet.optimizer.lr = 0.0001
        model_resnet.fit(train_x_resnet, train_y, epochs=100,
                         callbacks=[early_stopper])

        # Train model on optical flow features
        print("Training model on optical flow")
        model_optical_flow = conv_model(input_shape=optical_flow_features[0].shape)
        model_optical_flow.optimizer.lr = 0.0001
        model_optical_flow.fit(train_x_optical_flow, train_y, epochs=100,
                               callbacks=[early_stopper])

        # Make predictions on training data
        pred_resnet = model_resnet.predict(train_x_resnet)
        pred_optical_flow = model_optical_flow.predict(train_x_optical_flow)
        concatenated_features = np.hstack([pred_resnet, pred_optical_flow])

        # Train SVM on predictions from convolutional networks
        print("Training SVM")
        svm = LinearSVC()
        clf = GridSearchCV(svm, param_grid={'C': svm_parameters})
        clf.fit(concatenated_features, train_y_svm)

        # Evaluate models
        pred_resnet = model_resnet.predict(test_x_resnet)
        pred_optical_flow = model_optical_flow.predict(test_x_optical_flow)
        concatenated_features = np.hstack([pred_resnet, pred_optical_flow])


        resnet_scores.append(model_resnet.evaluate(test_x_resnet, test_y))
        optical_flow_scores.append(model_optical_flow.evaluate(test_x_optical_flow, test_y))
        svm_scores.append(accuracy_score(test_y_svm, clf.predict(concatenated_features)))

        split += 1
    
    resnet_scores = np.asarray(resnet_scores)
    mean_acc = np.mean(resnet_scores[:,1])
    std_acc = np.std(resnet_scores[:,1])
    print("Mean accuracy resnet: {}\n STD: {}".format(mean_acc, std_acc))

    optical_flow_scores = np.asarray(optical_flow_scores)
    mean_acc = np.mean(optical_flow_scores[:,1])
    std_acc = np.std(optical_flow_scores[:,1])
    print("Mean accuracy optical flow: {}\n STD: {}".format(mean_acc, std_acc))

    svm_scores = np.asarray(svm_scores)
    mean_acc = np.mean(svm_scores)
    std_acc = np.std(svm_scores)
    print("Mean accuracy svm: {}\n STD: {}".format(mean_acc, std_acc))