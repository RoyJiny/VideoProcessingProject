from stabilization import *
from background_subtraction import *
from matting import *
from tracking import *

import time
import json

ID1 = "315029405"
ID2 = "207880865"

OUTPUT_DIR = '../Outputs/'
TMP_DIR = '../Temp/'

def run(initial_video_path, background_img):
    print("Starting to Process Video")

    stabilized_video = f'{OUTPUT_DIR}stabilized_{ID1}_{ID2}.avi'
    extracted_video = f'{OUTPUT_DIR}extracted_{ID1}_{ID2}.avi'
    binary_mask_video = f'{OUTPUT_DIR}binary_{ID1}_{ID2}.avi'
    matting_video = f'{OUTPUT_DIR}matted_{ID1}_{ID2}.avi'
    alpha_video = f'{OUTPUT_DIR}alpha_{ID1}_{ID2}.avi'
    output_video = f'{OUTPUT_DIR}OUTPUT_{ID1}_{ID2}.avi'

    binary_frames_path = f'{TMP_DIR}binary_frames.np'
    time_metric_start = time.time()

    print('\nRunning video stabilization')
    apply_stabilization(initial_video_path, stabilized_video)
    stabilization_time = time.time() - time_metric_start

    print('\nRunning video background subtraction')
    binary_mask_frames = apply_background_mask(stabilized_video, extracted_video, binary_mask_video, binary_frames_path)
    background_sub_time = time.time() - time_metric_start

    print('\nRunning video matting')
    apply_matting(stabilized_video, binary_mask_video, background_img, matting_video, alpha_video)
    matting_time = time.time() - time_metric_start

    print('\nRunning video tracking')
    apply_tracking(matting_video, output_video, binary_mask_frames)
    tracking_time = time.time() - time_metric_start


    # NOTE: alpha and matted are created together - so concidered as same timings
    print('\nCreating timing.json\n')
    timing_data = {
        "time_to_stabilize": stabilization_time,
        "time_to_binary": background_sub_time,
        "time_to_alpha": matting_time,
        "time_to_matted": matting_time,
        "time_to_output": tracking_time
    }
    json.dump(timing_data, open(f'{OUTPUT_DIR}timing.json','w'))
    print("Timing Data:",json.dumps(timing_data,indent=2))


if __name__ == "__main__":
    run('../Inputs/INPUT.avi', '../Inputs/background.jpg')