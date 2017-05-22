import cv2
import numpy as np

from .features import convert_color, get_hog_features, \
    get_bin_spatial_features, get_color_hist_features


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes

    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    all_bboxes = list()
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((int(np.min(nonzerox)), int(np.min(nonzeroy))), (int(np.max(nonzerox)), int(np.max(nonzeroy))))
        # Draw the box on the image
        all_bboxes.append(bbox)

    for bbox in all_bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, clf, scaler, cspace='RGB', spatial=True, chist=True, hog=True,
              spatial_size=(32, 32), hist_bins=16, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
              vis=True):

    bboxes = list()

    if vis:
        draw_img = np.copy(img)

    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, cspace=cspace)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hogs = [hog1, hog2, hog3]

    for xb in range(nxsteps):
        for yb in range(nysteps):
            test_features = list()
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            if spatial:
                spatial_features = get_bin_spatial_features(subimg, size=spatial_size)
                test_features.append(spatial_features)

            if chist:
                _, _, _, _, hist_features = get_color_hist_features(subimg, nbins=hist_bins)
                test_features.append(hist_features)

            if hog:
                # Extract HOG for this patch
                hog_features = list()
                if hog_channel == 'ALL':
                    for hogch in hogs:
                        hog_feat = hogch[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_features.append(hog_feat)
                    hog_features = np.hstack(hog_features)
                else:
                    hog_features = hogs[hog_channel][ypos:ypos + nblocks_per_window,
                                   xpos:xpos + nblocks_per_window].ravel()

                test_features.append(hog_features)

            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack(test_features).reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                if vis:
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

                bbox = [(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)]

                bboxes.append(bbox)

    if vis:
        return draw_img, bboxes
    else:
        return bboxes
