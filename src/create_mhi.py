import cv2
import numpy as np
import os
import argparse

def create_mhi(video_file_path):
    """
    Converts a video to a single 
    grayscale image representing
    motion history.
    """
    video = cv2.VideoCapture(video_file_path)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

    motion_history = np.zeros((int(height), int(width)), np.float32)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(
            history=10, backgroundRatio=0.5, nmixtures=2)
    
    timestamp = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        cv2.motempl.updateMotionHistory(
                fgmask, motion_history, timestamp, duration=30)
        timestamp += 1

    video.release()

    return motion_history



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
            help="Path to a directory with data.")
    ap.add_argument("--output_dir", required=True,
            help="Path to save motion history images")
    args = vars(ap.parse_args())

    input_dir = args['input_dir']
    output_dir = args['output_dir']
    for subdir in os.listdir(input_dir):
        for ind, file_name in enumerate(
                os.listdir(os.path.join(input_dir, subdir))):
            file_path = os.path.join(input_dir, subdir, file_name)
            mhi = create_mhi(file_path)
            
            output_file = '{}_{}.jpg'.format(subdir, ind)
            save_path = os.path.join(output_dir, output_file)
            cv2.imwrite(save_path, mhi)


