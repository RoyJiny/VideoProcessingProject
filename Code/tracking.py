import numpy as np
import sys
from utils import *
import json

def apply_tracking(input_video, output_video, binary_frames_path):
    original_frames = extract_frames_list(input_video)
    binary_mask_frames = np.load(binary_frames_path, allow_pickle=True)
    
    output_frames = []
    tracking_coordiantes = []
    for frame_count in range(len(binary_mask_frames)):
        sys.stdout.write(f"--Applying tracking on frame: {frame_count+1}/{len(binary_mask_frames)}\r")
        sys.stdout.flush()
        
        y1,x1,y2,x2 = calculate_rectangle_coordinates(binary_mask_frames[frame_count])
        tracking_coordiantes.append([int((x1+x2)//2), int((y1+y2)//2), int(abs(x2-x1)//2), int(abs(y2-y1)//2)])
        
        output_frame = original_frames[frame_count]
        output_frame[x1,y1:y2] = [0,0,255]
        output_frame[x2,y1:y2] = [0,0,255]
        output_frame[x1:x2,y1] = [0,0,255]
        output_frame[x1:x2,y2] = [0,0,255]
        output_frames.append(output_frame)
    print("")

    write_video(output_frames, output_video, input_video)

    print("\nCreating tracking.json")
    tracking_json = {}
    for i, coordinates in enumerate(tracking_coordiantes):
        tracking_json[i] = coordinates
    json.dump(tracking_json, open('../Outputs/tracking.json','w'))
    return
