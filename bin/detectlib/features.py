import os

import cv2
import matplotlib.image as mpimg
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.utils import shuffle


def convert_color(img, cspace='YCrCb'):
    img = img.copy()
    if cspace != 'RGB':
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = eval('cv2.cvtColor(img, cv2.COLOR_RGB2{})'.format(cspace))
    return img


def get_color_hist_features(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


def get_bin_spatial_features(img, size=(32, 32)):
    feature_image = np.copy(img)
    features = cv2.resize(feature_image, size).ravel()
    return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call get_bin_spatial_features() and get_color_hist_features()
def _extract_features(file, cspace='RGB', spatial=True, chist=True, hog=True,
                      spatial_size=(32, 32), hist_bins=32,
                      orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0
                      ):
    fvec = list()
    # Read in each one by one
    image = mpimg.imread(file)
    image = image * 255

    # apply color conversion if other than 'RGB'
    feature_image = convert_color(image, cspace)
    # print('image.shape', image.shape, 'maxval', image.max())
    # print('fimage.shape', feature_image.shape, 'maxval', feature_image.max())

    # Apply get_bin_spatial_features() to get spatial color features
    if spatial:
        spatial_features = get_bin_spatial_features(feature_image, size=spatial_size)
        fvec.append(spatial_features)

    # Apply get_color_hist_features() also with a color space option now
    if chist:
        _, _, _, _, hist_features = get_color_hist_features(feature_image, nbins=hist_bins)
        fvec.append(hist_features)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog:

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        fvec.append(hog_features)

    fvec = np.concatenate(fvec)
    fvec = fvec.astype(np.float64)
    return fvec


def get_samples():
    base = os.path.abspath(__file__)
    base = base.split('/bin/')[0]

    imgs = list()

    counter = 0
    y_labels = list()

    for subdir, dirs, files in os.walk(os.path.join(base, 'data/vehicles')):
        for filename in files:

            filename = os.path.join(subdir, filename)

            if not filename.endswith('png'): continue

            # img = mpimg.imread(filename)
            imgs.append(filename)
            y_labels.append(1)
            counter += 1

    counter = 0
    for subdir, dirs, files in os.walk(os.path.join(base, 'data/non-vehicles')):
        for filename in files:

            filename = os.path.join(subdir, filename)

            if not filename.endswith('png'): continue

            # img = mpimg.imread(filename)
            imgs.append(filename)
            y_labels.append(0)
            counter += 1

    return shuffle(imgs, np.array(y_labels))


def extract_features(imgs, cspace='RGB', spatial=True, chist=True, hog=True,
                     spatial_size=(32, 32), hist_bins=16, hist_range=(0, 256),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     n_jobs=1
                     ):
    features = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_extract_features)(file, cspace, spatial, chist, hog,
                                                                              spatial_size, hist_bins,
                                                                              orient, pix_per_cell, cell_per_block,
                                                                              hog_channel) for file in imgs)

    features = np.vstack(features)
    return features


if __name__ == '__main__':
    pass
