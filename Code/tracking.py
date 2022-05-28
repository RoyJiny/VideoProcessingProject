import numpy.matlib as matlib
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from utils import *

# SET NUMBER OF PARTICLES
N = 100

def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior
    # add the velocity from the previous step
    state_drifted[:,0] += s_prior[:,4]
    state_drifted[:,1] += s_prior[:,5]
    
    # noise
    state_drifted[:,0] += np.round(np.random.normal(0, 2, size=(N,)))
    state_drifted[:,1] += np.round(np.random.normal(0, 2, size=(N,)))
    state_drifted[:,4] += np.round(np.random.normal(0, 0.75, size=(N,)))
    state_drifted[:,5] += np.round(np.random.normal(0, 0.75, size=(N,)))
    
    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16,16,16))

    x_start = state[0] - state[2]
    x_end = state[0] + state[2]
    y_start = state[1] - state[3]
    y_end = state[1] + state[3]
    
    patch = image[y_start:y_end, x_start:x_end]
    patch = patch // 16
    

    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            hist[patch[i,j][0],patch[i,j][1],patch[i,j][2]] += 1

    hist = hist.reshape((4096,1))
    
    # normalize
    hist = hist/np.sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)
    
    for i in range(N):
        r = np.random.uniform(low=0,high=1,size=1)[0]
        j = 0
        while cdf[j] < r:
            j += 1
        S_next[i] = previous_state[j]

    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    exp = np.sum(np.sqrt(p * q)) * 20
    return np.exp(exp)

def show_particles_on_frame(image: np.ndarray, state: np.ndarray, W: np.ndarray):
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    # Avg particle box
    # use weights vector W to create a weighted average
    (x_avg, y_avg, w_avg, h_avg) = (
        np.sum(W*state[:,0]),
        np.sum(W*state[:,1]),
        np.sum(W*state[:,2]),
        np.sum(W*state[:,3])
    )
    (x_avg, y_avg, w_avg, h_avg) = (
        x_avg - w_avg,
        y_avg - h_avg,
        2 * w_avg,
        2 * h_avg
    )

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = (
        state[np.argmax(W)][0],
        state[np.argmax(W)][1],
        state[np.argmax(W)][2],
        state[np.argmax(W)][3]
    )
    (x_max, y_max, w_max, h_max) = (
        x_max - w_max,
        y_max - h_max,
        2 * w_max,
        2 * h_max
    )
    
    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def calculate_rectangle_coordinates(frame):
    smoothed_mask = ndimage.median_filter(frame, 20)
    inter = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    out = np.zeros(smoothed_mask.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(smoothed_mask, out)
    # plt.imshow(out, cmap='gray')
    # plt.show()
    # exit()

    dim0_indicator = np.max(out, axis=0)
    dim0_idx = np.where(dim0_indicator == 1)[0]
    dim1_indicator = np.max(out, axis=1)
    dim1_idx = np.where(dim1_indicator == 1)[0]
    dim0_start = dim0_idx[0]
    dim0_end = dim0_idx[-1]
    dim1_start = dim1_idx[0]
    dim1_end = dim1_idx[-1]

    return (dim0_start, dim1_start, dim0_end-dim0_start, dim1_end-dim1_start)

def calculate_initial_state(binary_mask_frames):
    first_frame_coordinates = calculate_rectangle_coordinates(binary_mask_frames[0])
    second_frame_coordinates = calculate_rectangle_coordinates(binary_mask_frames[1])

    x,y,w,h = first_frame_coordinates
    x2,y2,w2,h2 = second_frame_coordinates
    x_center = x + w//2
    y_center = y + h//2

    return (x_center,y_center,w//2,h//2,x2-x,y2-y)

def apply_tracking(input_video, output_video, binary_mask_frames):
    capture = cv2.VideoCapture(input_video)
    params = get_video_parameters(capture)
    output = cv2.VideoWriter(output_video, params["fourcc"], params["fps"], (params["width"],params["height"]), True)
    
    if not capture.isOpened():
        print("failed to open capture for", input_video)
        return

    frame_count = 0
    while True:
        sys.stdout.write(f"Working on frame: {frame_count}\r")
        sys.stdout.flush()
        binary_frames = binary_mask_frames[frame_count:frame_count+2]
        state = calculate_initial_state(binary_frames)
        

        x,y,w,h = state[:4]
        (x, y, w, h) = (
            x - w,
            y - h,
            2 * w,
            2 * h
        )

        output_frame = frame
        output_frame[y,x:x+w] = [0,255,0]
        output_frame[y+h,x:x+w] = [0,255,0]
        output_frame[y:y+h,x] = [0,255,0]
        output_frame[y+h,x+w] = [0,255,0]
        output.write(output_frame)
    capture.release()
    output.release()
    cv2.destroyAllWindows()
    exit()


    s_initial = calculate_initial_state(binary_mask_frames)
    state_at_first_frame = matlib.repmat(s_initial, N, 1)
    S = predict_particles(state_at_first_frame)


    ret, frame = capture.read()
    if frame is None:
        return

    q = compute_normalized_histogram(frame, s_initial)

    W = np.zeros(N)
    for i in range(N):
        p = compute_normalized_histogram(frame, S[i])
        W[i] = bhattacharyya_distance(p,q)
    
    W = W/np.sum(W)

    C = np.cumsum(W)

    frame_count = 0
    from datetime import datetime
    start_time = datetime.now()
    while True:
        sys.stdout.write(f"Working on frame: {frame_count} ({datetime.now()-start_time})\r")
        sys.stdout.flush()
        frame_count += 1
        ret, frame = capture.read()
        if frame is None:
            break

        S_prev = S        
        S_next_tag = sample_particles(S_prev, C)
        S = predict_particles(S_next_tag)

        W = np.zeros(N)
        for i in range(N):
            p = compute_normalized_histogram(frame, S[i])
            W[i] = bhattacharyya_distance(p,q)
        
        W = W/np.sum(W)
        C = np.cumsum(W)

        output_frame = show_particles_on_frame(frame, S, W)
        output.write(output_frame)


    capture.release()
    output.release()
    cv2.destroyAllWindows()
