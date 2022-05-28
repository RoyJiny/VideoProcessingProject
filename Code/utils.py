import cv2

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

def extract_first_frame(input_video):
    capture = cv2.VideoCapture(input_video)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        break
    
    capture.release()
    cv2.destroyAllWindows()
    return frame

def write_video(frames, path, params_like):
    tmp_capture = cv2.VideoCapture(params_like)
    params = get_video_parameters(tmp_capture)
    output = cv2.VideoWriter(path, params["fourcc"], params["fps"], (params["width"],params["height"]), True)
    for frame in frames:
        output.write(frame)
    tmp_capture.release()
    output.release()
    cv2.destroyAllWindows()
