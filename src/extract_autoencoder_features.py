import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import argparse
import numpy as np
import h5py

def custom_sigmoid(x):
    return K.sigmoid(x) * 100


def huber_loss(y_true, y_pred, delta=1):
    """
    According to answer here:
    https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
    huber loss with dela=1 is very similar to MAE and is differentiable.
    """

    return tf.losses.huber_loss(y_true,y_pred, delta)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoencoder", required=True,
            help="Path the autoencoder.")
    ap.add_argument("--input_file", required=True,
            help="Path to a file with resnet50 features")
    ap.add_argument("--output_file", required=True,
            help="Path to save HDF5 file.")
    args = vars(ap.parse_args())
    

    model = load_model(args['autoencoder'],
                       custom_objects={'custom_sigmoid': custom_sigmoid,
                                        'huber_loss': huber_loss})

    autoencoder_layer = model.get_layer(name='encoded_common').output
    autoencoder = Model(inputs=model.inputs, outputs=autoencoder_layer)

    data = h5py.File(args['input_file'], 'r')
    resnet_features = data['resnet50'][:]
    resnet_features = np.expand_dims(resnet_features, axis=1)
    labels = data['labels'][:]
    data.close()
    
    autoencoder_features = np.zeros(shape=(len(resnet_features), 2048))
    
    #FastText representation corresponding to no text
    textual_data_vec = np.zeros(shape=(1, 10, 300))
    for i, resnet_feature in enumerate(resnet_features):
        if i % 10 == 0:
            print("Processed {} videos".format(i))
        encoded_vec = autoencoder.predict([textual_data_vec, resnet_feature])
        autoencoder_features[i] = np.squeeze(encoded_vec)


    output_file = h5py.File(args['output_file'], 'w')
    output_file.create_dataset('labels', data=np.array(labels, dtype='S'))
    output_file.create_dataset('autoencoder_features', data=autoencoder_features)
    output_file.close()
    
