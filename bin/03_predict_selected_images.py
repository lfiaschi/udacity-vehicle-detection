from detectlib.features import get_samples, get_hog_features, convert_color
from detectlib.tracker import CarTracker
import matplotlib.image as mpimg
import matplotlib.pylab as plt
import cv2
import numpy as np

tracker = CarTracker('model.p',
                     nsteps=3,
                     threshold=4,
                     scales=[0.8, 1.5, 3, 6],
                     minsize_detection_close=3000,
                     minsize_detection_far = 2500
                     )


count = 0

import glob

# for i in range(1, 7):
#      img = mpimg.imread('test_images/test{}.jpg'.format(i))
#
#     res = tracker.process_image(img)
#     heat = tracker.get_current_heatmap(img)
#
#     print('Heatmap Max', heat.max())
#
#
#     fig = plt.figure(figsize=(12, 8))
#     plt.subplot(131)
#
#     plt.imshow(img)
#
#     plt.subplot(132)
#     plt.imshow(heat)
#
#     plt.subplot(133)
#     plt.imshow(res)
#
#     plt.show()

n = 0
for i in sorted(glob.glob('short_video_frames/*.png')):
    n+=1
    if n< 40 :
        continue
    # img = mpimg.imread('test_images/test{}.jpg'.format(i))
    img = mpimg.imread(i)*255

    img = img.astype(np.uint8)

    heat = tracker.get_current_heatmap(img)
    heat_integrated = tracker._integrate_heatmap(heat)
    res = tracker._treshold_and_label(img, heat_integrated)

    fig = plt.figure(figsize=(12, 8))
    plt.subplot(141)
    plt.imshow(img)

    plt.subplot(142)
    plt.imshow(heat)

    plt.subplot(143)
    plt.imshow(heat_integrated)

    plt.subplot(144)
    plt.imshow(res)

    plt.show()
