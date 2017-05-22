from detectlib.tracker import *
import pickle
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os

import sys

white_output = sys.argv[2]
input = sys.argv[1]

print('Input file ', input)
print('Output file', white_output)

clip1 = VideoFileClip(input)



# tracker = CarTracker('model.p',
#                      nsteps=3,
#                      threshold=4,
#                      scales=[0.8, 1, 1.5, 2],
#                      minsize_detection=800
#                      )

tracker = CarTracker('model.p',
                     nsteps=3,
                     threshold=4,
                     scales=[0.8,1, 1.5, 3, 6],
                     minsize_detection_close=2800,
                     minsize_detection_far=500
                     )


# clip1.write_images_sequence('short_video_frames/frame%03d.png')
white_clip = clip1.fl_image(tracker.process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
