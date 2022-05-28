import cv2
import numpy as np
import sys

from utils import *

SMOOTHING_RADIUS = 5

def moving_avg(series, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    series_pad = np.lib.pad(series, (radius, radius), 'reflect')
    series_smoothed = np.convolve(series_pad, f, mode='same')
    series_smoothed = series_smoothed[radius:-radius]
    return series_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:,i] = moving_avg(trajectory[:,i],SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1],s[0]))
    return frame

def apply_stabilization(input_video, output_video):
    frames = extract_frames_list(input_video)
    gray_first_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create()
    prev_key_points, prev_descriptors = detector.detectAndCompute(gray_first_frame, None)

    transforms = np.zeros((len(frames)-1, 9), dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    for i,frame in enumerate(frames[1:]):
        sys.stdout.write(f"--Calculating Transform for frame: {i+2}/{len(frames)}\r")
        sys.stdout.flush()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = detector.detectAndCompute(gray_frame, None)

        matches = bf.match(prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda m: m.distance)[:int(len(matches)*0.15)]

        src_points = np.float32([prev_key_points[m.queryIdx].pt for m in matches]).reshape(-1,2)
        dst_points = np.float32([key_points[m.trainIdx].pt for m in matches]).reshape(-1,2)
        
        M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

        transforms[i] = M.flatten()
        prev_key_points, prev_descriptors = key_points, descriptors
    print("")

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    diff = smoothed_trajectory - trajectory
    smoothed_transforms = transforms + diff
    h,w = frames[0].shape[:2]
    stabilized_frames = [frames[0]]
    
    for i, frame in enumerate(frames[:-1]):
        sys.stdout.write(f"--Stabilizing frame: {i+2}/{len(frames)}\r")
        sys.stdout.flush()
        M = smoothed_transforms[i].reshape((3,3))
        out = cv2.warpPerspective(frame, M, (w,h))
        out = fixBorder(out)
        stabilized_frames.append(out)
    print("")

    write_video(stabilized_frames, output_video, input_video)
