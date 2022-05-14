from stabilization import *
from background_subtraction import *
from matting import *
from tracking import *

ID1 = "315029405"
ID2 = "207880865"

OUTPUT_DIR = '../Outputs/'

def run(initial_video_path, background_img):
    stabilized_video = f'{OUTPUT_DIR}stabilized_{ID1}_{ID2}.avi'
    extracted_video = f'{OUTPUT_DIR}extracted_{ID1}_{ID2}.avi'
    binary_mask_video = f'{OUTPUT_DIR}binary_{ID1}_{ID2}.avi'
    matting_video = f'{OUTPUT_DIR}matted_{ID1}_{ID2}.avi'
    output_video = f'{OUTPUT_DIR}OUTPUT_{ID1}_{ID2}.avi'

    print('Running video stabilization')
    # lucas_kanade_faster_video_stabilization(initial_video_path, stabilized_video)
    apply_stabilization(initial_video_path, stabilized_video)

    print('Running video background subtraction')
    binary_mask_frames = apply_background_mask(initial_video_path, extracted_video, binary_mask_video)

    print('Running video matting')
    # apply_matting(extracted_video, background_img, binary_mask_video, matting_video)

    print('Running video tracking')
    apply_tracking(matting_video, output_video, binary_mask_frames)

if __name__ == "__main__":
    run('../Inputs/INPUT.avi', '../Inputs/background.jpg')