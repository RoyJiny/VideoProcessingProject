import cv2
import sys
import matplotlib.pyplot as plt

from utils import *

video = sys.argv[1]
only_first_frame = 'first' in sys.argv
cap = cv2.VideoCapture(video)
params = get_video_parameters(cap)
print(f'Total of {params["frame_count"]} frames')
if (cap.isOpened()== False): 
  print("Error opening video  file")
   
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    if only_first_frame:
      plt.imshow(frame)
      plt.show()
      break
   
    cv2.imshow('Frame', frame)
   
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
   
  else: 
    break
   
cap.release()
   
cv2.destroyAllWindows()