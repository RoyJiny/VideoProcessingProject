from utils import *

import numpy as np
import cv2

SMOOTHING_RADIUS = 3

def moving_avg(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius,radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:,i] = moving_avg(trajectory[:,i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1],s[0]))
    return frame

def apply_stabilization(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    params = get_video_parameters(cap)
    n_frames = params["frame_count"]
    w = params["width"]
    h = params["height"]
    fourcc = params["fourcc"]
    output = cv2.VideoWriter(output_video, fourcc, params["fps"], (w,h))

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in range(n_frames - 2):
        prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        ret,frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None)
        assert prev_points.shape == curr_points.shape

        idx = np.where(status == 1)[0]
        prev_points = prev_points[idx]
        curr_points = curr_points[idx]

        m = cv2.estimateAffinePartial2D(prev_points, curr_points)[0]

        dx = m[0,2]
        dy = m[1,2]

        da = np.arctan2(m[1,0], m[0,0])

        transforms[i] = [dx,dy,da]

        prev_gray = frame_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames-2):
        ret, frame = cap.read()
        if not ret:
            break

        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]

        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (w,h))
        frame_stabilized = fixBorder(frame_stabilized)
       
        output.write(frame_stabilized)
