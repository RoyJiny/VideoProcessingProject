import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

from utils import *

video = f"../Outputs/{sys.argv[1]}_315029405_207880865.avi"
# video = f"../Outputs/{sys.argv[1]}_roy_315029405_207880865.avi"
# video = "masks_tmp.avi"
input_vid = "../Inputs/INPUT.avi"
only_first_frame = 'first' in sys.argv

input_cap = cv2.VideoCapture(input_vid)
cap = cv2.VideoCapture(video)
params = get_video_parameters(cap)
print(f'Total of {params["frame_count"]} frames')
   
while True:
  input_ret, input_frame = input_cap.read()
  ret, frame = cap.read()
  if ret == True and input_ret == True:
    if only_first_frame:
      plt.imshow(frame)
      plt.show()
      break

    cv2.imshow("frame", np.concatenate((input_frame, frame)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
   
  else: 
    break
   
input_cap.release()   
cap.release()
cv2.destroyAllWindows()