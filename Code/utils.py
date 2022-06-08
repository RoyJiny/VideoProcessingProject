import cv2
import numpy as np

def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}

def extract_frames_list(input_video):
    capture = cv2.VideoCapture(input_video)
    frames = []
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frames.append(frame)
    
    capture.release()
    cv2.destroyAllWindows()
    return frames

def write_video(frames, path, params_like='../Inputs/INPUT.avi'):
    tmp_capture = cv2.VideoCapture(params_like)
    params = get_video_parameters(tmp_capture)
    output = cv2.VideoWriter(path, params["fourcc"], params["fps"], (params["width"],params["height"]), True)
    for frame in frames:
        output.write(frame.astype(np.uint8))
    tmp_capture.release()
    print(f"Saved Video at {path}")
    output.release()
    cv2.destroyAllWindows()

def calculate_rectangle_coordinates(frame, delta=0):
    if len(np.unique(frame)) <= 1:
        print(f'Bad binary frame, holding values: {np.unique(frame)}')
    
    dim0_indicator = np.max(frame, axis=0)
    dim0_idx = np.where(dim0_indicator == np.unique(frame)[1])[0]
    dim1_indicator = np.max(frame, axis=1)
    dim1_idx = np.where(dim1_indicator == np.unique(frame)[1])[0]
    
    dim0_start = max(dim0_idx[0]-delta,0)
    dim0_end = min(dim0_idx[-1]+delta, frame.shape[1]-1)
    dim1_start = max(dim1_idx[0]-delta, 0)
    dim1_end = min(dim1_idx[-1]+delta, frame.shape[0]-1)

    return (dim0_start, dim1_start, dim0_end, dim1_end)
