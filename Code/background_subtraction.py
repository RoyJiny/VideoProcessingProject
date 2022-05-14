import cv2
import numpy as np
from utils import *

def calculate_binary_mask(frames):
    backsub = cv2.createBackgroundSubtractorMOG2()
    binary_mask_frames = []
    for frame in frames:
        foreground_mask = backsub.apply(frame)
        binary_mask = (foreground_mask == 255).astype(np.uint8)
        binary_mask_frames.append(binary_mask)
    return binary_mask_frames

def apply_background_mask(input_video, output_video, mask_output_video):
    params = get_video_parameters(cv2.VideoCapture(input_video))
    
    output = cv2.VideoWriter(output_video, params["fourcc"], params["fps"], (params["width"],params["height"]), True)
    mask_output = cv2.VideoWriter(mask_output_video, params["fourcc"], params["fps"], (params["width"],params["height"]), True)
    
    frames = extract_frames_list(input_video)
    binary_mask_frames_normal = calculate_binary_mask(frames)
    binary_mask_frames_reversed = list(reversed(calculate_binary_mask(list(reversed(frames)))))
    num_of_frames = len(frames)
    binary_mask_frames = binary_mask_frames_reversed[:num_of_frames//2] + binary_mask_frames_normal[num_of_frames//2:]
    
    for frame,binary_mask in zip(frames,binary_mask_frames):
        binary_mask = np.dstack((binary_mask,binary_mask,binary_mask))
        output.write(frame * binary_mask)
        mask_output.write(binary_mask*255)

    output.release()
    mask_output.release()
    cv2.destroyAllWindows()

    return binary_mask_frames