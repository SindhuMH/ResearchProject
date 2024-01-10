import numpy as np
import cv2
import copy
from skimage.filters import roberts
from collections import Counter
from operator import itemgetter
from utilities import Utilities


class ImageProcessor:
    def __init__(self, img_size=400):
        """
        Initializes an instance of the ImageSegmentation class.

        Parameters:
        - img_size (int): Size to which the images will be resized.
        """
        self.img_size = img_size

    @staticmethod
    def binarize_and_extract_largest_region(original_image):
        """
        Binarizes the image using adaptive thresholding and extracts the largest connected component.

        Parameters:
        - original_image: Input image.

        Returns:
        - extracted_region: Image with the largest connected component.
        - largest_component_mask: Mask of the largest connected component.
        """
        _, binary_image = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

        # Find the index of the largest connected component (excluding background)
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create a mask for the largest connected component
        largest_component_mask = (labels == largest_component_index).astype(np.uint8)

        # Extract the region from the original image using the mask
        extracted_region = cv2.bitwise_and(original_image, original_image, mask=largest_component_mask)

        return extracted_region, largest_component_mask

    @staticmethod
    def adaptive_clahe(img, power_fact=2):
        """
        Applies adaptive contrast-limited histogram equalization (CLAHE) to the image.

        Parameters:
        - img: Input image.
        - power_fact (int): Power factor for the adaptive CLAHE.

        Returns:
        - new_img: Processed image.
        """
        intense_dict = dict()
        Dm = np.max(img)
        vec_im = img.flatten()
        N = len(vec_im)
        c = Counter(vec_im)
        sort_vec = sorted(c.items(), key=itemgetter(0))
        x, y = img.shape
        new_im = np.zeros((x, y))

        prev_ele = None
        for tup in sort_vec:
            i = tup[0]
            n = tup[1]
            tot_n = n
            if prev_ele is None:
                intense_dict[i] = tot_n
                prev_ele = i
            else:
                tot_n = tot_n + intense_dict[prev_ele]
                intense_dict[i] = tot_n
                prev_ele = i
        i = 0
        j = 0

        for Ai in vec_im:
            n = intense_dict.get(Ai)
            Fi = ((n / N) ** power_fact) * (Ai + (n / N * ((n * Dm / n) - Ai)))
            new_im[i][j] = Fi

            j = j + 1
            if j == y:
                j = 0
                i = i + 1
        min_int = min(new_im.flatten())
        new_im = new_im - min_int
        return new_im

    @staticmethod
    def grayscale_erosion_and_reconstruction(original_image, dit=1, eit=1):
        """
        Applies grayscale erosion and reconstruction to the input image.

        Parameters:
        - original_image: Input image.
        - dit (int): Number of dilations.
        - eit (int): Number of erosions.

        Returns:
        - enhanced: Enhanced image after erosion and reconstruction.
        """
        thresh = copy.deepcopy(original_image)
        threshd = cv2.dilate(thresh, None, iterations=dit)
        threshe = cv2.erode(thresh, None, iterations=eit)

        outline = threshd - threshe
        enhanced = thresh - outline

        return enhanced

    @staticmethod
    def roberts_edge(image):
        """
        Applies the Roberts edge detection to the input image.

        Parameters:
        - image: Input image.

        Returns:
        - edges_roberts: Image edges detected using Roberts filter.
        """
        edges_roberts = roberts(image)
        return edges_roberts

    def preprocess_image(self, path):
        """
        Preprocesses the input image by applying binarization, adaptive CLAHE, and masking.

        Parameters:
        - path (str): Path to the input image.
        - ksize (int): Size of the adaptive CLAHE kernel.
        - no_resize (bool): If True, the image will not be resized.

        Returns:
        - aimg: Preprocessed image.
        """
        img = Utilities.read_image(path, no_resize=True)
        bin_img, mask = self.binarize_and_extract_largest_region(img)
        aimg = self.adaptive_clahe(bin_img, 10)
        aimg[mask == 0] = -1
        return aimg

    @staticmethod
    def reconstruct_im(labels, image_shape, window_size):
        """
        Reconstructs an image from labeled regions.

        Parameters:
        - labels: List of labels for each region.
        - image_shape: Shape of the output image.
        - window_size: Size of the regions.

        Returns:
        - image: Reconstructed image.
        """
        num_rows, num_cols = (image_shape, image_shape)
        image = np.zeros((num_rows, num_cols), dtype=labels.dtype)

        for i, label in enumerate(labels):
            row = i // (num_cols // window_size)
            col = i % (num_cols // window_size)
            start_row = row * window_size
            start_col = col * window_size
            vector = np.asarray([label] * (window_size * window_size))
            window = vector.reshape((window_size, window_size))
            image[start_row:start_row + window_size, start_col:start_col + window_size] = window
        return image

    @staticmethod
    def get_windows(full_im, window_size):
        """
        Generates non-overlapping windows from the input image.

        Parameters:
        - full_im: Input image.
        - window_size: Size of the windows.

        Returns:
        - imgartrain: List of vectorized windows.
        - img_dict: Dictionary indicating if a window has at least one non-zero pixel.
        """
        # Initialize the arrays and dictionary
        imgartrain = []
        img_dict = {}
        img_size = full_im.shape[0]

        im_vec_ind = 0
        # Iterate through the image with non-overlapping windows
        for i in range(0, img_size, window_size):
            for j in range(0, img_size, window_size):
                window = full_im[i:i + window_size, j:j + window_size]
                # Check if the window has at least one non-zero pixel
                if np.any(window != -1):
                    img_dict[im_vec_ind] = 1
                    imgartrain.append(window)  # Append vectorized window to imgartrain
                else:
                    img_dict[im_vec_ind] = -1

                im_vec_ind = im_vec_ind + 1
        return imgartrain, img_dict

    def calculate_data(self, image, window_size):
        """
        Calculates data for training a model.

        Parameters:
        - image: Input image.
        - window_size: Size of the windows.
        - gradient: Whether to include gradient information.

        Returns:
        - data: Stacked vectors of windows.
        - img_dict: Dictionary indicating if a window has at least one non-zero pixel.
        """
        windows, img_dict = self.get_windows(image, window_size)
        rows = len(windows)
        if rows == 0:
            return None, None
        i = 0
        vectors = []
        for window in windows:
            vector = window.flatten()
            vectors.append(vector)
            i = i + 1
        return np.vstack(vectors), img_dict

    def segment_im(self, ind, window_size, gmm, df, plot_ims=False):
        """
        Segments an image using a Gaussian Mixture Model (GMM).

        Parameters:
        - ind: Index of the image.
        - window_size: Size of the windows.
        - ksize: Size of the adaptive CLAHE kernel.
        - gmm: Gaussian Mixture Model.
        - df: DataFrame containing image information.
        - plot_ims: Whether to plot the images.
        - gradient: Whether to include gradient information.

        Returns:
        - clustered_img: Segmented image.
        """
        rf = df.loc[ind]['reference_number']
        path = "data/images_aug/" + rf + ".pgm"
        full_img = self.preprocess_image(path)

        data, img_dict = self.calculate_data(full_img, window_size)
        pred = gmm.predict(data)
        pred_ar = []
        pred_ind = 0

        for i in img_dict:
            if img_dict[i] == 1:
                pred_ar.append(pred[pred_ind])
                pred_ind = pred_ind + 1

            else:
                pred_ar.append(-1)

        pred_ar = np.asarray(pred_ar)
        clustered_img = self.reconstruct_im(pred_ar, full_img.shape[0], window_size)

        if plot_ims:
            Utilities.plot_array(1, 2, (full_img, clustered_img))
        return clustered_img
