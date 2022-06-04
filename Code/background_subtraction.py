import cv2
import numpy as np
from scipy import ndimage
from utils import *
import sys
import matplotlib.pyplot as plt

def clean_binary_frame(frame):
    smoothed_mask = ndimage.median_filter(frame*255, 21)
    inter = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    out = np.zeros(smoothed_mask.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(smoothed_mask, out)
    return out

def clean_and_cut_masks(frames, masks):
    new_frames = []
    for i,(frame,mask) in enumerate(zip(frames,masks)):
        sys.stdout.write(f"--Cleaning and cutting mask: {i+1}/{len(frames)}\r")
        sys.stdout.flush()
        cleaned_mask = clean_binary_frame(mask)
        x1,y1,x2,y2 = calculate_rectangle_coordinates(cleaned_mask)
        new_mask = np.zeros(mask.shape)
        new_mask[y1:,x1:x2] = 1
        new_frames.append((frame * new_mask).astype(np.uint8))
    print("")
    return new_frames


def calculate_binary_mask(frames):
    backsub_mog = cv2.createBackgroundSubtractorMOG2()
    binary_mask_frames = []
    for i,frame in enumerate(frames):
        sys.stdout.write(f"--MOG for frame: {i+1}/{len(frames)}\r")
        sys.stdout.flush()
        foreground_mask_mog = (backsub_mog.apply(frame) == 255).astype(np.uint8)
        binary_mask_frames.append(foreground_mask_mog)
    print("")
    return binary_mask_frames

def apply_mog_and_knn(frames):
    backsub_mog = cv2.createBackgroundSubtractorMOG2()
    backsub_knn = cv2.createBackgroundSubtractorKNN()
    binary_mask_frames = []
    for i,frame in enumerate(frames):
        sys.stdout.write(f"--MOG & KNN for frame: {i+1}/{len(frames)}\r")
        sys.stdout.flush()
        foreground_mask_mog = (backsub_mog.apply(frame) == 255)
        foreground_mask_knn = (backsub_knn.apply(frame) == 255)
        binary_mask = np.logical_or(foreground_mask_mog,foreground_mask_knn).astype(np.uint8)
        binary_mask_frames.append(binary_mask)   
    print("") 
    
    return binary_mask_frames

def apply_background_mask(input_video, output_video, mask_output_video, binary_frames_path):
    frames = extract_frames_list(input_video)
    num_of_frames = len(frames)

    binary_mask_frames_normal = apply_mog_and_knn(frames)
    binary_mask_frames_reversed = list(reversed(apply_mog_and_knn(list(reversed(frames)))))
    binary_mask_frames = binary_mask_frames_reversed[:num_of_frames//2] + binary_mask_frames_normal[num_of_frames//2:]

    window_binary_mask_frames_normal = calculate_binary_mask(frames)
    window_binary_mask_frames_reversed = list(reversed(calculate_binary_mask(list(reversed(frames)))))
    window_binary_mask_frames = window_binary_mask_frames_reversed[:num_of_frames//2] + window_binary_mask_frames_normal[num_of_frames//2:]
    
    binary_mask_frames = clean_and_cut_masks(binary_mask_frames, window_binary_mask_frames)

    for i in range(len(binary_mask_frames)):
        sys.stdout.write(f"--Cleaning mask for frame: {i+1}/{len(frames)}\r")
        sys.stdout.flush()
        binary_mask_frames[i] = clean_binary_frame(binary_mask_frames[i])
    print("")

    output_frames = []
    mask_output_frames = []
    for i,(frame,binary_mask) in enumerate(zip(frames,binary_mask_frames)):
        sys.stdout.write(f"--Masking frame: {i+1}/{len(frames)}\r")
        sys.stdout.flush()
        binary_mask = np.dstack((binary_mask,binary_mask,binary_mask))
        output_frames.append(frame * (binary_mask//255))
        mask_output_frames.append(binary_mask)
    print("")

    write_video(output_frames, output_video, input_video)
    write_video(mask_output_frames, mask_output_video, input_video)

    np.array(binary_mask_frames).dump(binary_frames_path)