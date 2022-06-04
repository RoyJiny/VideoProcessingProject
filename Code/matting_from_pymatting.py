import GeodisTK
import cv2
import numpy as np
import logging
import sys

'''VIDEO STABILIZATION CONST.'''
MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 30
BLOCK_SIZE = 3
SMOOTH_RADIUS = 5

'''BACKGROUND SUBTRACTION CONST.'''
BW_MEDIUM = 1
BW_NARROW = 0.1
LEGS_HEIGHT = 805
SHOES_HEIGHT = 870
SHOULDERS_HEIGHT = 405
BLUE_MASK_THR = 140
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 1000
FACE_WINDOW_HEIGHT = 250
FACE_WINDOW_WIDTH = 300

'''MATTING CONST.'''
EPSILON_NARROW_BAND = 0.99
ERODE_ITERATIONS = 6
DILATE_ITERATIONS = 5
GEODISTK_ITERATIONS = 1
REFINEMENT_WINDOW_SIZE = 20
KDE_BW = 1
R = 2



def check_in_dict(dict, element, function):
    if element in dict:
        return dict[element]
    else:
        dict[element] = function(np.asarray(element))[0]
        return dict[element]


import numpy as np
import cv2
from scipy.stats import gaussian_kde


def fixBorder(frame):
    h, w = frame.shape[0],frame.shape[1]
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame


def get_video_files(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height, fps


def release_video_files(cap):
    cap.release()
    cv2.destroyAllWindows()


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'reflect')

    '''Fix padding manually'''
    for i in range(radius):
        curve_pad[i] = curve_pad[radius] - curve_pad[i]

    for i in range(len(curve_pad) - 1, len(curve_pad) - 1 - radius, -1):
        curve_pad[i] = curve_pad[len(curve_pad) - radius - 1] - curve_pad[i]

    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth(trajectory, smooth_radius):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=smooth_radius)
    return smoothed_trajectory


def write_video(output_path, frames, fps, out_size, is_color):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(output_path, fourcc, fps, out_size, isColor=is_color)
    for frame in frames:
        video_out.write(frame)
    video_out.release()


def scale_matrix_0_to_255(input_matrix):
    if input_matrix.dtype == np.bool:
        input_matrix = np.uint8(input_matrix)
    input_matrix = input_matrix.astype(np.uint8)
    scaled = 255 * (input_matrix - np.min(input_matrix)) / np.ptp(input_matrix)
    return np.uint8(scaled)


def load_entire_video(cap, color_space='bgr'):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        if color_space == 'bgr':
            frames.append(curr)
        elif color_space == 'yuv':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2YUV))
        elif color_space == 'bw':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)


def apply_mask_on_color_frame(frame, mask):
    frame_after_mask = np.copy(frame)
    frame_after_mask[:, :, 0] = frame_after_mask[:, :, 0] * mask
    frame_after_mask[:, :, 1] = frame_after_mask[:, :, 1] * mask
    frame_after_mask[:, :, 2] = frame_after_mask[:, :, 2] * mask
    return frame_after_mask


def choose_indices_for_foreground(mask, number_of_choices):
    indices = np.where(mask == 1)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0],indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))


def choose_indices_for_background(mask, number_of_choices):
    indices = np.where(mask == 0)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0],indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))

def new_estimate_pdf(omega_values, bw_method):
    pdf = gaussian_kde(omega_values.T, bw_method=bw_method)
    return lambda x: pdf(x.T)

def estimate_pdf(original_frame, indices, bw_method):
    omega_f_values = original_frame[indices[:, 0], indices[:, 1], :]
    pdf = gaussian_kde(omega_f_values.T, bw_method=bw_method)
    return lambda x: pdf(x.T)


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))

# font = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (10, 50)
# fontScale = 3
# fontColor = (255, 255, 255)
# lineType = 2
#
# cv2.putText(weighted_mask, str(i),
#             bottomLeftCornerOfText,
#             font,
#             fontScale,
#             fontColor,
#             lineType)
#
# cv2.imshow('s',weighted_mask)
# cv2.waitKey(0)

# # Write the frame to the file
# concat_frame = cv2.hconcat([mask_or, mask_or_erosion])
# # If the image is too big, resize it.
# if concat_frame.shape[1] > 1920:
#     concat_frame = cv2.resize(concat_frame, (int(concat_frame.shape[1]), int(concat_frame.shape[0])))
# cv2.imshow("Before and After", concat_frame)
# cv2.waitKey(0)

# image = np.copy(frame_after_or_and_blue_flt)
# for index in range(chosen_pixels_indices.shape[0]):
#     image = cv2.circle(image, (chosen_pixels_indices[index][1], chosen_pixels_indices[index][0]), 5, (0, 255, 0), 2)
# Displaying the image
# cv2.imshow('sas', image)
# cv2.waitKey(0)



my_logger = logging.getLogger('MyLogger')


def video_matting(input_stabilize_video, binary_video_path, new_background, output_video, alpha_video):
    my_logger.info('Starting Matting')

    # Read input video
    cap_stabilize, w, h, fps_stabilize = get_video_files(path=input_stabilize_video)
    cap_binary, _, _, fps_binary = get_video_files(path=binary_video_path)

    # Get frame count
    n_frames = int(cap_stabilize.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')
    frames_yuv = load_entire_video(cap_stabilize, color_space='yuv')
    frames_binary = load_entire_video(cap_binary, color_space='bw')

    '''Resize new background'''
    new_background = cv2.imread(new_background)
    new_background = cv2.resize(new_background, (w, h))

    '''Starting Matting Process'''
    full_matted_frames_list, alpha_frames_list = [], []
    for frame_index in range(n_frames):
        sys.stdout.write(f'[Matting] - Frame: {frame_index} / {n_frames}\r')
        sys.stdout.flush()
        luma_frame, _, _ = cv2.split(frames_yuv[frame_index])
        bgr_frame = frames_bgr[frame_index]

        original_mask_frame = frames_binary[frame_index]
        original_mask_frame = (original_mask_frame > 150).astype(np.uint8)

        '''Find indices for resizing image to work only on relevant part!'''
        DELTA = 20
        binary_frame_rectangle_x_axis = np.where(original_mask_frame == 1)[1]
        left_index, right_index = np.min(binary_frame_rectangle_x_axis), np.max(binary_frame_rectangle_x_axis)
        left_index, right_index = max(0, left_index - DELTA), min(right_index + DELTA, original_mask_frame.shape[1] - 1)
        binary_frame_rectangle_y_axis = np.where(original_mask_frame == 1)[0]
        top_index, bottom_index = np.min(binary_frame_rectangle_y_axis), np.max(binary_frame_rectangle_y_axis)
        top_index, bottom_index = 0,original_mask_frame.shape[0] - 1
        # top_index, bottom_index = max(0, top_index - DELTA), min(bottom_index + DELTA, original_mask_frame.shape[0] - 1)

        ''' Resize images '''
        smaller_luma_frame = luma_frame[top_index:bottom_index, left_index:right_index]
        smaller_bgr_frame = bgr_frame[top_index:bottom_index, left_index:right_index]
        smaller_new_background = new_background[top_index:bottom_index, left_index:right_index]

        '''Erode & Resize foreground mask & Build distance map for foreground'''
        foreground_mask = cv2.erode(original_mask_frame, np.ones((3, 3)), iterations=ERODE_ITERATIONS)
        smaller_foreground_mask = foreground_mask[top_index:bottom_index, left_index:right_index]
        smaller_foreground_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_foreground_mask,
                                                                          1.0, GEODISTK_ITERATIONS)

        '''Dilate & Resize image & Build distance map for background'''
        background_mask = cv2.dilate(original_mask_frame, np.ones((3, 3)), iterations=DILATE_ITERATIONS)
        background_mask = 1 - background_mask
        smaller_background_mask = background_mask[top_index:bottom_index, left_index:right_index]
        smaller_background_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_background_mask,
                                                                          1.0, GEODISTK_ITERATIONS)

        ''' Building narrow band undecided zone'''
        smaller_foreground_distance_map = smaller_foreground_distance_map / (smaller_foreground_distance_map + smaller_background_distance_map)
        smaller_background_distance_map = 1 - smaller_foreground_distance_map
        smaller_narrow_band_mask = (np.abs(smaller_foreground_distance_map - smaller_background_distance_map) < EPSILON_NARROW_BAND).astype(np.uint8)
        smaller_narrow_band_mask_indices = np.where(smaller_narrow_band_mask == 1)

        smaller_decided_foreground_mask = (smaller_foreground_distance_map < smaller_background_distance_map - EPSILON_NARROW_BAND).astype(np.uint8)
        smaller_decided_background_mask = (smaller_background_distance_map >= smaller_foreground_distance_map - EPSILON_NARROW_BAND).astype(np.uint8)
        '''Building KDEs for foreground & background to calculate priors for alpha calculation'''
        omega_f_indices = choose_indices_for_foreground(smaller_decided_foreground_mask, 200)
        omega_b_indices = choose_indices_for_background(smaller_decided_background_mask, 200)
        foreground_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_f_indices, bw_method=KDE_BW)
        background_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_b_indices, bw_method=KDE_BW)
        smaller_narrow_band_foreground_probs = foreground_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])
        smaller_narrow_band_background_probs = background_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])

        '''Start creating alpha map'''
        w_f = np.power(smaller_foreground_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_foreground_probs
        w_b = np.power(smaller_background_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_background_probs
        alpha_narrow_band = w_f / (w_f + w_b)
        smaller_alpha = np.copy(smaller_decided_foreground_mask).astype(np.float)
        smaller_alpha[smaller_narrow_band_mask_indices] = alpha_narrow_band

        '''Naive implementation for matting as described in algorithm'''
        smaller_matted_frame = smaller_alpha[:, :, np.newaxis] * smaller_bgr_frame + (1 - smaller_alpha[:, :, np.newaxis]) * smaller_new_background

        '''move from small rectangle to original size'''
        full_matted_frame = np.copy(new_background)
        full_matted_frame[top_index:bottom_index, left_index:right_index] = smaller_matted_frame
        full_matted_frames_list.append(full_matted_frame)

        full_alpha_frame = np.zeros(original_mask_frame.shape)
        full_alpha_frame[top_index:bottom_index, left_index:right_index] = smaller_alpha
        full_alpha_frame = (full_alpha_frame * 255).astype(np.uint8)
        alpha_frames_list.append(full_alpha_frame)

    write_video(output_path=output_video, frames=full_matted_frames_list, fps=fps_stabilize, out_size=(w, h),
                is_color=True)
    write_video(output_path=alpha_video, frames=alpha_frames_list, fps=fps_stabilize, out_size=(w, h), is_color=False)
    print('~~~~~~~~~~~ [Matting] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ matted.avi has been created! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ alpha.avi has been created! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ unstabilized_alpha.avi has been created! ~~~~~~~~~~~')
    my_logger.info('Finished Matting')

