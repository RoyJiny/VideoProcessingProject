import cv2
import numpy as np
from utils import *

def apply_matting(input_video, background_img, mask_video, output_video):
    capture = cv2.VideoCapture(input_video)
    mask_capture = cv2.VideoCapture(mask_video)
    
    background = cv2.imread(background_img)
    
    params = get_video_parameters(capture)
    output = cv2.VideoWriter(output_video, params["fourcc"], params["fps"], (params["width"],params["height"]), True)
    
    if not capture.isOpened():
        print("failed to open capture for", input_video)
        return
    if not mask_capture.isOpened():
        print("failed to open capture for", mask_video)
        return
    
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        ret, mask_frame = mask_capture.read()
        if mask_frame is None:
            break
        
        binary_mask = (mask_frame > 127).astype(np.uint8)
        output_frame = binary_mask * frame + (1 - binary_mask) * background
        output.write(output_frame)

    capture.release()
    mask_capture.release()
    output.release()
    cv2.destroyAllWindows()
