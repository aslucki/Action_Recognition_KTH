import cv2
import numpy as np
import pims
import os
import h5py
import argparse

def process_frame(frame):
    frame = np.asarray(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    return frame


def scale_flow(flow):
    negative_indices = flow<0
    flow[negative_indices] *= -255
    flow[~negative_indices] *= 255

    return flow.astype(int)


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



    input_dir = args['input_dir']
    n_samples = count_files(input_dir)
    L = 10

    features = np.zeros(shape=(n_samples, 120, 160, 2*L))
    
    labels = []
    ind = 0
    for subdir in os.listdir(input_dir):
        for file_name in os.listdir(os.path.join(input_dir, subdir)):
            file_path = os.path.join(input_dir, subdir, file_name)

            video = pims.PyAVReaderIndexed(file_path)
            selected_frames = video[:L+1]

            flows = []
            for prev_frame, next_frame in zip(selected_frames, selected_frames[1:]):
                prev_frame = process_frame(prev_frame)
                next_frame = process_frame(next_frame)
                flow = cv2.calcOpticalFlowFarneback(prev_frame,
                                        next_frame,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = scale_flow(flow)
                flows.append(flow)

            flows = np.asarray(flows)
            flows = np.concatenate(flows, axis=2)


            labels.append(subdir)
            features[ind,...,:flows.shape[-1]] = flows
            ind += 1
            print(ind/n_samples*100)

    output_file = h5py.File(args['output_file'], 'w')
    output_file.create_dataset('labels', data=np.array(labels, dtype='S'))
    output_file.create_dataset('optical_flow', data=features)
    output_file.close()
