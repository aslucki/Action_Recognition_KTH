import os
from skimage import feature
import mahotas
import cv2
import numpy as np
import h5py
import argparse

def extract_hog(image, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), transform_sqrt=True,
                block_norm='L2-Hys'):
    features = feature.hog(image, orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           transform_sqrt=transform_sqrt,
                           block_norm=block_norm)
    return features

def extract_zernike(image, radius=100, degree=18):
    features = mahotas.features.zernike_moments(
                                    image, radius=radius,
                                    degree=degree)
    return features

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
            help="Path to a directory with data.")
    ap.add_argument("--output_file", required=True,
            help="Path to save HDF5 file with features.")
    args = vars(ap.parse_args())

    hog_features = []
    zernike_moments = []
    labels = []
    input_dir = args['input_dir']
    
    for file_name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, file_name)
        mhi = cv2.imread(full_path)
        mhi = cv2.cvtColor(mhi, cv2.COLOR_BGR2GRAY)
        label = file_name.split('_')[0]
        
        hog_features.append(extract_hog(mhi))
        zernike_moments.append(extract_zernike(mhi))
        labels.append(label)

    output_file = h5py.File(args['output_file'], 'a')
    output_file.create_dataset('labels', data=np.array(labels, dtype='S'))
    output_file.create_dataset('hog', data=hog_features)
    output_file.create_dataset('zernike', data=zernike_moments)

    output_file.close()
