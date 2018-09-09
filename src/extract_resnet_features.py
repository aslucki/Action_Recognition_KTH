import pims
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras import Model
import cv2
import os
import h5py
import argparse

def process_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    converted = np.array(resized_frame, dtype='float64')

    return preprocess_input(converted)

def count_files(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len(files)

    return total

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
            help="Path to a directory with data.")
    ap.add_argument("--output_file", required=True,
            help="Path to save HDF5 file.")
    args = vars(ap.parse_args())

    MAX_FRAME_NUM = 250
    FRAMES_INTERVAL = 25

    model = ResNet50(weights='imagenet', include_top=False)
    
    input_dir = args['input_dir']
    n_samples = count_files(input_dir)
    n_timesteps = int(MAX_FRAME_NUM / FRAMES_INTERVAL)

    features = np.zeros(shape=(n_samples, n_timesteps, 2048))
    
    labels = []
    ind = 0
    for subdir in os.listdir(input_dir):
        for file_name in os.listdir(os.path.join(input_dir, subdir)):
            file_path = os.path.join(input_dir, subdir, file_name)

            video = pims.PyAVReaderIndexed(file_path)
            selected_frames = video[:MAX_FRAME_NUM:FRAMES_INTERVAL]

            sequential_features = np.zeros(shape=(n_timesteps, 2048))
            for i, frame in enumerate(selected_frames):
                processed = process_frame(frame)

                preds = model.predict(np.expand_dims(processed, axis=0))
                sequential_features[i] = np.squeeze(preds)

            labels.append(subdir)
            features[ind] = sequential_features
            ind += 1

    output_file = h5py.File(args['output_file'], 'w')
    output_file.create_dataset('labels', data=np.array(labels, dtype='S'))
    output_file.create_dataset('resnet50', data=features)
    output_file.close()
