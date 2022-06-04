from stabilization import *
from background_subtraction import *
from matting import *
from tracking import *

ID1 = "315029405"
ID2 = "207880865"

OUTPUT_DIR = '../Outputs/'
TMP_DIR = '../Temp/'

def run(initial_video_path, background_img):
    tmp_video = f'{OUTPUT_DIR}tmp.avi'
    stabilized_video = f'{OUTPUT_DIR}stabilized_{ID1}_{ID2}.avi'
    extracted_video = f'{OUTPUT_DIR}extracted_{ID1}_{ID2}.avi'
    binary_mask_video = f'{OUTPUT_DIR}binary_{ID1}_{ID2}.avi'
    matting_video = f'{OUTPUT_DIR}matted_{ID1}_{ID2}.avi'
    alpha_video = f'{OUTPUT_DIR}alpha_{ID1}_{ID2}.avi'
    output_video = f'{OUTPUT_DIR}OUTPUT_{ID1}_{ID2}.avi'

    binary_frames_path = f'{TMP_DIR}binary_frames.np'

    # print('\nRunning video stabilization')
    # apply_stabilization(initial_video_path, stabilized_video)

    print('\nRunning video background subtraction')
    apply_background_mask(stabilized_video, extracted_video, binary_mask_video, binary_frames_path)

    # print('\nRunning video matting')
    # apply_matting(extracted_video, background_img, binary_mask_video, matting_video)
    # from matting_from_pymatting import video_matting
    # video_matting(stabilized_video, binary_mask_video, background_img, matting_video, alpha_video)

    # print('\nRunning video tracking')
    # apply_tracking(matting_video, output_video, binary_frames_path)


if __name__ == "__main__":
    run('../Inputs/INPUT.avi', '../Inputs/background.jpg')