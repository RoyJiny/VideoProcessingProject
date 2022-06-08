import GeodisTK
import cv2
from scipy.stats import gaussian_kde
import numpy as np
import sys

from utils import *

def get_mask_indices(mask, number_of_choices, value):
    indices = np.where(mask == value)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0],indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))

def estimate_pdf(original_frame, indices):
    omega_f_values = original_frame[indices[:, 0], indices[:, 1], :]
    pdf = gaussian_kde(omega_f_values.T, bw_method=1)
    return lambda x: pdf(x.T)

def apply_matting(input_video, binary_video, background_path, output_video, output_alpha_video):
    frames = extract_frames_list(input_video)
    binary_frames = extract_frames_list(binary_video)
    binary_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in binary_frames]
    yuv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2YUV) for frame in frames]

    background = cv2.imread(background_path)

    matted_frames = []
    alpha_frames = []

    for i,(frame,binary_frame,yuv_frame) in enumerate(zip(frames,binary_frames,yuv_frames)):
        sys.stdout.write(f"--Matting frame: {i+1}/{len(frames)}\r")
        sys.stdout.flush()

        luma_channel = cv2.split(yuv_frame)[0]
        binary_frame = (binary_frame > 127).astype(np.uint8)

        x1,y1,x2,y2 = calculate_rectangle_coordinates(binary_frame, delta=20)
        y1 = 0
        y2 = frame.shape[0] - 1

        ''' create masks '''
        foreground_mask = cv2.erode(binary_frame, np.ones((3,3)), iterations=6)
        background_mask = 1 - cv2.dilate(binary_frame, np.ones((3,3)), iterations=5)

        ''' crop to interst area '''
        cropped_luma = luma_channel[y1:y2,x1:x2]
        cropped_frame = frame[y1:y2,x1:x2]
        cropped_background = background[y1:y2,x1:x2]
        cropped_foreground_mask = foreground_mask[y1:y2,x1:x2]
        cropped_background_mask = background_mask[y1:y2,x1:x2]

        ''' calculate distance maps '''
        cropped_foreground_mask_dist_map = GeodisTK.geodesic2d_raster_scan(cropped_luma, cropped_foreground_mask, 1.0, 1)
        cropped_background_mask_dist_map = GeodisTK.geodesic2d_raster_scan(cropped_luma, cropped_background_mask, 1.0, 1)

        ''' create narrow band undecided zone '''
        cropped_foreground_mask_dist_map = cropped_foreground_mask_dist_map / (cropped_foreground_mask_dist_map + cropped_background_mask_dist_map)
        cropped_background_mask_dist_map = 1 - cropped_foreground_mask_dist_map
        cropped_narrow_band_mask = (np.abs(cropped_foreground_mask_dist_map - cropped_background_mask_dist_map) < 0.99).astype(np.uint8)
        cropped_narrow_band_mask_indices = np.where(cropped_narrow_band_mask == 1)

        cropped_decided_foreground_mask = (cropped_foreground_mask_dist_map < cropped_background_mask_dist_map - 0.99).astype(np.uint8)
        cropped_decided_background_mask = (cropped_background_mask_dist_map >= cropped_foreground_mask_dist_map - 0.99).astype(np.uint8)
        
        ''' build KDE for background and foreground '''
        omega_f_indices = get_mask_indices(cropped_decided_foreground_mask, 200, 1)
        omega_b_indices = get_mask_indices(cropped_decided_background_mask, 200, 0)
        foreground_pdf = estimate_pdf(cropped_frame, omega_f_indices)
        background_pdf = estimate_pdf(cropped_frame, omega_b_indices)
        cropped_narrow_band_foreground_probs = foreground_pdf(cropped_frame[cropped_narrow_band_mask_indices])
        cropped_narrow_band_background_probs = background_pdf(cropped_frame[cropped_narrow_band_mask_indices])

        ''' create alpha map '''
        w_f = np.power(cropped_foreground_mask_dist_map[cropped_narrow_band_mask_indices],-2) * cropped_narrow_band_foreground_probs
        w_b = np.power(cropped_background_mask_dist_map[cropped_narrow_band_mask_indices],-2) * cropped_narrow_band_background_probs
        alpha_narrow_band = w_f / (w_f + w_b)
        cropped_alpha = np.copy(cropped_decided_foreground_mask).astype(np.float)
        cropped_alpha[cropped_narrow_band_mask_indices] = alpha_narrow_band

        ''' apply matting to the cropped frame '''
        cropped_matted_frame = cropped_alpha[:, :, np.newaxis] * cropped_frame + (1 - cropped_alpha[:, :, np.newaxis]) * cropped_background

        ''' place the cropped frame on the background to get back to the original size '''
        matted_frame = np.copy(background)
        matted_frame[y1:y2,x1:x2] = cropped_matted_frame
        matted_frames.append(matted_frame)

        alpha_frame = np.zeros(binary_frame.shape)
        alpha_frame[y1:y2,x1:x2] = cropped_alpha
        alpha_frame = (alpha_frame * 255).astype(np.uint8)
        alpha_frames.append(np.dstack((alpha_frame,alpha_frame,alpha_frame)))
    print("")

    write_video(matted_frames, output_video)
    write_video(alpha_frames, output_alpha_video)