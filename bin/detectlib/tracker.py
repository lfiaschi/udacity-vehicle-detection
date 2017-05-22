from .sliding_windows import *
from joblib import Parallel,delayed
import pickle
from skimage.filters import gaussian
from scipy.ndimage import label

# Search at at a single scale, keep hear to be able to parallelize the search
def _search_scale(img,scale, self):

    assert img.max() == 255

    feat_img = img.copy()
    bboxes = find_cars(feat_img, self.ystart, self.ystop, scale, **self.options, vis=False)

    return bboxes

# Used to treshold out detections which are too small, implements two different levels for close or far away detections
def remove_small_detections(labels, minsize_close=3000, minsize_far=2000):

    mapped  = labels[0]

    for i in range(1,labels[1]+1):
        nonzero = (mapped == i).nonzero()
        nonzeroy = np.max(np.array(nonzero[0]))

        size = np.sum(mapped==i)
        print(size)
        # Threshold out small detections which are close or far with two different thresholds
        if size < minsize_close and nonzeroy>450 or size< minsize_far and nonzeroy<=450:
            mapped[mapped==i] = 0

    return label(mapped)


class CarTracker(object):

    def __init__(self, model_file,
                 scales = [0.5, 1, 1.5, 2, 2.5, 3],
                 nsteps = 5,
                 threshold = 5,
                 vertical_roi=(400,656),
                 minsize_detection_close=np.inf,
                 minsize_detection_far = np.inf,
                 ):
        """
        Main class to perform detection on video
        :param train_clf_file: path to the output of the training script
        :param scales: scales to perform the search
        :param nsteps: keep the last nsteps and average across time, 0 to disable
        :param threshold: threshold confidence for the detection
        :param vertical_roi: (ystart,ystop) define a region of interest for the detection and the sliding window search
        :param minsize_detection_close= treshold for the size of far close detections (close objects appear bigger)
        :param minsize_detection_far = treshold for the size of far awy detections
        """
        self.model_file = model_file

        with open(model_file,'rb') as fh:
            self.options = pickle.load(fh)

        self.scales = scales
        self.nsteps = nsteps

        self.treshold = threshold

        self.ystart = vertical_roi[0]
        self.ystop = vertical_roi[1]

        #Keep a running list of last n heatmaps
        self._heatmaps = list()

        self.ncals = 0

        self.minsize_detection_close = minsize_detection_close
        self.minsize_detection_far = minsize_detection_far

    def process_image(self, img):

        assert img.max() == 255 and img.dtype == np.uint8 #assert that the image is correctly an rgb image

        self.ncals=+1

        current_heat = self.get_current_heatmap(img)

        heatmap = self._integrate_heatmap(current_heat)

        return self._treshold_and_label(img, heatmap)

    def get_current_heatmap(self, img):
        all_bboxes = list()

        # Search all scales in parallel
        tmp = Parallel(n_jobs=len(self.scales))(delayed(_search_scale)(img,scale,self) for scale in self.scales)

        [all_bboxes.extend(bboxes) for bboxes in tmp]

        # import ipdb; ipdb.set_trace()
        current_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        current_heat = add_heat(current_heat, all_bboxes)

        return current_heat

    # Integrate heatmap with past nsteps, smooth heatmap a little
    def _integrate_heatmap(self, current_heatmap):

        assert current_heatmap.ndim == 2

        if self._heatmaps and len(self._heatmaps) > self.nsteps:
            self._heatmaps.pop()

        self._heatmaps.append(current_heatmap)

        heatmap = np.dstack(self._heatmaps) if len(self._heatmaps) > 1 else current_heatmap

        print(heatmap.shape, heatmap.max())

        # Integrate Heatmap over time
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=-1)

            # Smooth a little the heatmaps
            heatmap = gaussian(heatmap, 4)

        return heatmap

    def _treshold_and_label(self, img, heatmap):

        heatmap[heatmap <= self.treshold] = 0

        heatmap = np.clip(heatmap, 0, 255)

        labels = label(heatmap)

        if self.minsize_detection_close:
            final_labels = remove_small_detections(labels, self.minsize_detection_close , self.minsize_detection_far)

        print('N cars detected ={}'.format(final_labels[1]))

        draw_img = draw_labeled_bboxes(np.copy(img), final_labels)

        return draw_img


