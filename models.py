import numpy as np
from tqdm import tqdm
import copy
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from image_preprocess import ImageProcessor
from extract_roi import RoiExtract
from utilities import Utilities


class Models:

    def __init__(self):
        self.img_processor = ImageProcessor()
        self.roi_ext = RoiExtract()

    def get_gmm(self, df, window_size, kmeans_model, num_components=3):
        """
        Fits and returns a Gaussian Mixture Model (GMM) based on the provided DataFrame.

        Parameters:
        - df: DataFrame containing image information.
        - window_size (int): Size of the windows.
        - kmeans_model: Trained KMeans model.
        - num_components (int): Number of components for the GMM.

        Returns:
        - gmm: Trained Gaussian Mixture Model.
        """
        roi1 = self.roi_ext.get_roi(df, window_size)
        roi2 = self.roi_ext.get_roi(df, window_size, is_norm=True)

        combined_rois = roi1 + roi2

        random.shuffle(combined_rois)

        data = []
        for roi in tqdm(combined_rois):
            windows, img_dict = ImageProcessor.get_windows(roi, window_size)
            vectors = []
            i = 0
            for window in windows:
                vector = window.flatten()
                vectors.append(vector)
                i = i + 1
            vecs = np.vstack(vectors)
            data.append(vecs)
        data = np.vstack(data)

        print("fitting gmm model", data.shape)

        gmm = GaussianMixture(n_components=num_components, covariance_type='full',
                              means_init=kmeans_model.cluster_centers_, max_iter=1000)
        gmm.fit(data)

        print("converged?", gmm.converged_)
        print("total iterations?", gmm.n_iter_)
        return gmm

    def get_preds(self, ind, df, target_clust, gmm, window_size, draw=False):
        """
        Generates predictions for a specific image using a trained GMM model.

        Parameters:
        - ind: Index of the image in the DataFrame.
        - df: DataFrame containing image information.
        - target_clust: Target cluster label.
        - gmm: Trained Gaussian Mixture Model.
        - window_size (int): Size of the windows.
        - ksize: Kernel size.
        - draw (bool): Whether to draw the predictions.

        Returns:
        - iou: Intersection over Union (IoU) score.
        """
        expected_circle = df.loc[ind][['ab_x', 'ab_y', 'ab_radius']].values.astype('int32')
        image_id = df.loc[ind]['reference_number']
        path = 'data/images_aug/' + image_id + '.pgm'
        img = Utilities.read_image(path, no_resize=True)
        aimg = self.img_processor.preprocess_image(path)

        clust_im = self.img_processor.segment_im(ind, window_size, gmm, df)
        image_shape = clust_im.shape

        clust_im = clust_im + 1
        max_label = np.max(clust_im)
        clust_im = np.round((clust_im / max_label) * 255).astype('uint8')
        target_clust = np.round((target_clust / max_label) * 255).astype('uint8')

        circles_info, contours_ = self.roi_ext.extract_fit_circles_gray(clust_im, target_clust)

        selected_circle, iou, contourr = self.roi_ext.find_max_iou_circle(circles_info, expected_circle, image_shape, contours_)

        if draw:
            cluster_mask = (clust_im == target_clust).astype(np.uint8)
            rois = copy.deepcopy(clust_im)

            if selected_circle is None:
                for x, y, r in circles_info:
                    rois = self.roi_ext.draw_circle(rois, x, y, r, 0, do_offset=False)

            else:
                x, y, r = selected_circle
                self.roi_ext.draw_circle(rois, x, y, r, 0, do_offset=False)

            x, y, r = expected_circle
            img = self.roi_ext.draw_circle(img, x, y, r, 0, do_offset=True)
            Utilities.plot_array(2, 2, [img, rois, aimg, cluster_mask])
        return iou

    def get_kmeans(self, dtr, num_clusters, window_size):
        """
        Retrieves a trained KMeans model and the target cluster label.

        Parameters:
        - dtr: DataFrame containing image information for training.
        - num_clusters (int): Number of clusters for KMeans.
        - window_size (int): Size of the windows.

        Returns:
        - kmeans: Trained KMeans model.
        - target_km: Target cluster label.
        """
        inds = dtr.loc[dtr['abnormality_type'] != 'NORM'].index.tolist()
        ind = inds[11]
        print(dtr.loc[ind])

        rf = dtr.loc[ind]['reference_number']
        path = "data/images_aug/" + rf + ".pgm"

        original_image = self.img_processor.preprocess_image(path)

        # Reshape the resized image into a 1D array (flattening)
        data, img_dict = self.img_processor.calculate_data(original_image, window_size)

        # Initialize and fit Kmeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(data)

        # Extract labels and reshape back to the resized image shape
        pred = kmeans.labels_

        pred_ar = []
        pred_ind = 0

        for i in img_dict:
            if img_dict[i] == 1:
                pred_ar.append(pred[pred_ind])
                pred_ind = pred_ind + 1

            else:
                pred_ar.append(-1)

        pred_ar = np.asarray(pred_ar)
        clustered_img = ImageProcessor.reconstruct_im(pred_ar, original_image.shape[0], window_size)

        Utilities.plot_array(1, 2, [original_image, clustered_img])

        target_km = input(f"please choose cluster label for abnormality between 1 to {num_clusters} : ")
        target_km = float(target_km)
        return kmeans, target_km

    def get_kmeans_preds(self, ind, df, window_size, target_clust, kmeans, draw=False):
        """
        Generates predictions for a specific image using a trained KMeans model.

        Parameters:
        - ind: Index of the image in the DataFrame.
        - df: DataFrame containing image information.
        - window_size (int): Size of the windows.
        - target_clust: Target cluster label.
        - kmeans: Trained KMeans model.
        - draw (bool): Whether to draw the predictions.

        Returns:
        - iou: Intersection over Union (IoU) score.
        """
        expected_circle = df.loc[ind][['ab_x', 'ab_y', 'ab_radius']].values.astype('int32')
        image_id = df.loc[ind]['reference_number']
        path = 'data/images_aug/' + image_id + '.pgm'
        img = Utilities.read_image(path, no_resize=True)

        original_image = self.img_processor.preprocess_image(path)
        data, img_dict = self.img_processor.calculate_data(original_image, window_size)
        pred = kmeans.predict(data)

        pred_ar = []
        pred_ind = 0

        for i in img_dict:
            if img_dict[i] == 1:
                pred_ar.append(pred[pred_ind])
                pred_ind = pred_ind + 1

            else:
                pred_ar.append(-1)

        pred_ar = np.asarray(pred_ar)
        clust_im = ImageProcessor.reconstruct_im(pred_ar, original_image.shape[0], window_size)

        image_shape = clust_im.shape

        clust_im = clust_im + 1
        max_label = np.max(clust_im)
        clust_im = np.round((clust_im / max_label) * 255).astype('uint8')
        target_clust = np.round((target_clust / max_label) * 255).astype('uint8')

        circles_info, contours_ = self.roi_ext.extract_fit_circles_gray(clust_im, target_clust)

        selected_circle, iou, contourr = self.roi_ext.find_max_iou_circle(circles_info, expected_circle, image_shape, contours_)

        if draw:
            cluster_mask = (clust_im == target_clust).astype(np.uint8)
            rois = copy.deepcopy(clust_im)

            if selected_circle is None:
                selected_circles = []
                ful_ar = img.shape[0] ** 2
                for x, y, r in circles_info:
                    area = np.pi * (r ** 2)
                    if 0.12 * ful_ar >= area >= 0.0005 * ful_ar:
                        selected_circles.append((x, y, r))

                for x, y, r in selected_circles:
                    rois = self.roi_ext.draw_circle(rois, x, y, r, 0, do_offset=False)

            else:
                x, y, r = selected_circle
                self.roi_ext.draw_circle(rois, x, y, r, 0, do_offset=False)

            x, y, r = expected_circle
            img = self.roi_ext.draw_circle(img, x, y, r, 0, do_offset=True)
            Utilities.plot_array(2, 2, [img, rois, clust_im, cluster_mask])
        return iou

    def get_kmeans_ious(self, dts, window_size, kmeans_, targetk):
        """
        Computes IoU scores for a set of images using a trained KMeans model.

        Parameters:
        - dts: DataFrame containing image information for testing.
        - window_size (int): Size of the windows.
        - kmeans_: Trained KMeans model.
        - targetk: Target cluster label.

        Returns:
        - kiou_scores_f: Dictionary of IoU scores for each image.
        """
        inds = dts.loc[dts['abnormality_type'] != 'NORM'].index.tolist()
        kiou_scores_f = dict()

        for ind in tqdm(inds):
            iou = self.get_kmeans_preds(ind, dts, window_size, targetk, kmeans_, False)
            kiou_scores_f[ind] = iou

        return kiou_scores_f
