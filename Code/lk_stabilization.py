import cv2
import numpy as np
import scipy
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata
from utils import *


# FILL IN YOUR ID
ID1 = 207880865
ID2 = 315029405


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def build_pyramid(image: np.ndarray, num_levels: int):
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """

    pyramid = [image.copy()]
    for i in range(num_levels):
        level = signal.convolve2d(pyramid[-1], PYRAMID_FILTER,  boundary='symm', mode='same')
        level = level[::2, ::2]
        pyramid.append(level)
    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int):
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    I1 = I1.astype(np.float64)
    I2 = I2.astype(np.float64)
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = np.subtract(I2, I1)
    w = window_size // 2
    for i in range(w, Ix.shape[0] - w):
        for j in range(w, Ix.shape[1] - w):
            x_window = (Ix[i - w:i + w+1, j - w:j + w+1]).flatten()
            y_window = (Iy[i - w:i + w+1, j - w:j + w+1]).flatten()
            t_window = (It[i - w:i + w+1, j - w:j + w+1]).flatten()
            A = np.column_stack((x_window, y_window))
            try:
                [u, v] = np.matmul(np.matmul(-np.linalg.inv(np.matmul(A.T, A)), A.T), t_window)
            except np.linalg.LinAlgError:
                u, v = 0, 0
            du[i, j] = u
            dv[i, j] = v
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image = image.astype(np.float64)
    u_cols = u.shape[1]
    v_rows = v.shape[0]
    u_ratio = image.shape[1] / u_cols
    v_ratio = image.shape[0] / v_rows
    u *= u_ratio
    v *= v_ratio
    u = cv2.resize(u, image.T.shape)
    v = cv2.resize(v, image.T.shape)
    grid_points = np.meshgrid(list(range(image.shape[1])), list(range(image.shape[0])))
    grid_points = (grid_points[0].flatten(), grid_points[1].flatten())
    grid_points_uv = (np.add(grid_points[0], u.flatten()), np.add(grid_points[1], v.flatten()))
    image_warp = scipy.interpolate.griddata(grid_points, image.flatten(), grid_points_uv, fill_value=np.nan)
    image_warp = image_warp.reshape(image.shape)
    image_warp[np.isnan(image_warp)] = image[np.isnan(image_warp)]
    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int):
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)
    for i in range(num_levels, -1, -1):
        warped_I2 = warp_image(pyramid_I2[i], u, v)
        for j in range(max_iter):
            du, dv = lucas_kanade_step(pyramid_I1[i], warped_I2, window_size)
            u += du
            v += dv
            warped_I2 = warp_image(pyramid_I2[i], u, v)
        if i > 0:
            u = cv2.resize(u, pyramid_I2[i-1].T.shape)*2
            v = cv2.resize(v, pyramid_I2[i-1].T.shape)*2
    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    """INSERT YOUR CODE HERE."""
    capture = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(capture)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_factor = int(np.ceil(frame.shape[0] / (2 ** (num_levels - 1 + 1))))
        w_factor = int(np.ceil(frame.shape[1] / (2 ** (num_levels - 1 + 1))))
        IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                      h_factor * (2 ** (num_levels - 1 + 1)))
        if frame.shape != IMAGE_SIZE:
            frame = cv2.resize(frame, IMAGE_SIZE)
        frames.append(frame)
    output = cv2.VideoWriter(output_video_path, fourcc, params["fps"], IMAGE_SIZE,
                             False)
    output.write(frames[0].astype(np.uint8))
    prev_u = np.zeros(frames[0].shape)
    prev_v = np.zeros(frames[0].shape)
    w = window_size // 2
    for i in tqdm(range(1, len(frames))):
        u, v = lucas_kanade_optical_flow(frames[i-1], frames[i], window_size, max_iter, num_levels)
        u_mean = u[w:u.shape[0] - w, w:u.shape[1] - w].mean()
        v_mean = v[w:v.shape[0] - w, w:v.shape[1] - w].mean()
        prev_u[w:u.shape[0] - w, w:u.shape[1] - w] += u_mean
        prev_v[w:v.shape[0] - w, w:v.shape[1] - w] += v_mean
        warp_frame = warp_image(frames[i], prev_u, prev_v)
        output.write(warp_frame.astype(np.uint8))

    capture.release()
    output.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int):
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    if I1.size < 25000:
        return lucas_kanade_step(I1, I2, window_size)
    dest = cv2.cornerHarris(I2.astype(np.uint8), 2, 5, 0.07)
    I, J = np.where(dest > 0.001 * dest.max())
    I1 = I1.astype(np.float64)
    I2 = I2.astype(np.float64)
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = np.subtract(I2, I1)
    w = window_size // 2
    for i, j in zip(I, J):
        if w <= i < Ix.shape[0] - w and w <= j < Ix.shape[1] - w:
            x_window = (Ix[i - w:i + w + 1, j - w:j + w + 1]).flatten()
            y_window = (Iy[i - w:i + w + 1, j - w:j + w + 1]).flatten()
            t_window = (It[i - w:i + w + 1, j - w:j + w + 1]).flatten()
            A = np.column_stack((x_window, y_window))
            try:
                [u, v] = np.matmul(np.matmul(-np.linalg.inv(np.matmul(A.T, A)), A.T), t_window)
            except np.linalg.LinAlgError:
                u, v = 0, 0
            du[i, j] = u
            dv[i, j] = v
    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int):
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)
    for i in range(num_levels, -1, -1):
        warped_I2 = warp_image(pyramid_I2[i], u, v)
        for j in range(max_iter):
            du, dv = faster_lucas_kanade_step(pyramid_I1[i], warped_I2, window_size)
            u += du
            v += dv
            warped_I2 = warp_image(pyramid_I2[i], u, v)
        if i > 0:
            u = cv2.resize(u, pyramid_I2[i - 1].T.shape) * 2
            v = cv2.resize(v, pyramid_I2[i - 1].T.shape) * 2
    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size:int = 5,
        max_iter:int = 5, num_levels:int = 5) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    capture = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(capture)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_factor = int(np.ceil(frame.shape[0] / (2 ** (num_levels - 1 + 1))))
        w_factor = int(np.ceil(frame.shape[1] / (2 ** (num_levels - 1 + 1))))
        IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                      h_factor * (2 ** (num_levels - 1 + 1)))
        if frame.shape != IMAGE_SIZE:
            frame = cv2.resize(frame, IMAGE_SIZE)
        frames.append(frame)
    output = cv2.VideoWriter(output_video_path, fourcc, params["fps"], IMAGE_SIZE,
                             False)
    output.write(frames[0].astype(np.uint8))
    prev_u = np.zeros(frames[0].shape)
    prev_v = np.zeros(frames[0].shape)
    w = window_size // 2
    for i in tqdm(range(1, len(frames))):
        u, v = faster_lucas_kanade_optical_flow(frames[i-1], frames[i], window_size, max_iter, num_levels)
        u_mean = u[w:u.shape[0] - w, w:u.shape[1] - w].mean()
        v_mean = v[w:v.shape[0] - w, w:v.shape[1] - w].mean()
        prev_u[w:u.shape[0] - w, w:u.shape[1] - w] += u_mean
        prev_v[w:v.shape[0] - w, w:v.shape[1] - w] += v_mean
        warp_frame = warp_image(frames[i], prev_u, prev_v)
        output.write(warp_frame.astype(np.uint8))

    capture.release()
    output.release()
    cv2.destroyAllWindows()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    capture = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(capture)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_factor = int(np.ceil(frame.shape[0] / (2 ** (num_levels - 1 + 1))))
        w_factor = int(np.ceil(frame.shape[1] / (2 ** (num_levels - 1 + 1))))
        IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                      h_factor * (2 ** (num_levels - 1 + 1)))
        if frame.shape != IMAGE_SIZE:
            frame = cv2.resize(frame, IMAGE_SIZE)
        frames.append(frame)
    output = cv2.VideoWriter(output_video_path, fourcc, params["fps"],
                             (IMAGE_SIZE[0]-start_cols-end_cols, IMAGE_SIZE[1]-start_rows-end_rows),
                             False)
    output.write(frames[0][start_rows:-end_rows, start_cols:-end_cols].astype(np.uint8))
    prev_u = np.zeros(frames[0].shape)
    prev_v = np.zeros(frames[0].shape)
    w = window_size // 2
    for i in tqdm(range(1, len(frames))):
        u, v = faster_lucas_kanade_optical_flow(frames[i-1], frames[i], window_size, max_iter, num_levels)
        u_mean = u[w:u.shape[0] - w, w:u.shape[1] - w].mean()
        v_mean = v[w:v.shape[0] - w, w:v.shape[1] - w].mean()
        prev_u[w:u.shape[0] - w, w:u.shape[1] - w] += u_mean
        prev_v[w:v.shape[0] - w, w:v.shape[1] - w] += v_mean
        warp_frame = warp_image(frames[i], prev_u, prev_v)
        warp_frame = warp_frame[start_rows:-end_rows, start_cols:-end_cols]
        output.write(warp_frame.astype(np.uint8))

    capture.release()
    output.release()
    cv2.destroyAllWindows()

