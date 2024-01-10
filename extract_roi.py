import numpy as np
from tqdm import tqdm
import math
import cv2
from image_preprocess import ImageProcessor


class RoiExtract:

    def __init__(self):
        self.img_processor = ImageProcessor()

    @staticmethod
    def create_circle_mask(center, radius, image_shape):
        """
        Creates a binary circular mask.

        Parameters:
        - center (tuple): Center coordinates (x, y) of the circle.
        - radius (int): Radius of the circle.
        - image_shape (tuple): Shape of the image.

        Returns:
        - mask: Binary circular mask.
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=cv2.FILLED)
        return mask

    @staticmethod
    def closest_multiple(number, window_size):
        """
        Rounds up the given number to the closest multiple of window_size.

        Parameters:
        - number (int): Number to be rounded.
        - window_size (int): Target window size.

        Returns:
        - multiple: Closest multiple of window_size.
        """
        remainder = number % window_size
        if remainder == 0:
            return number
        else:
            multiple = number // window_size + 1
        return multiple * window_size

    @staticmethod
    def draw_circle(image, x, y, r, c=255, do_offset=False):
        """
        Draws a circle on the given image.

        Parameters:
        - image: Input image.
        - x, y, r (int): Circle parameters (center coordinates and radius).
        - c (int): Circle color.
        - do_offset (bool): Whether to apply offset to the y-coordinate.

        Returns:
        - image: Image with the drawn circle.
        """
        h, w = image.shape
        if do_offset:
            image = cv2.circle(image, (x, h - y), r, c, 4)
        else:
            image = cv2.circle(image, (x, y), r, c, 4)
        return image

    @staticmethod
    def extract_fit_circles_gray(gray_image, target_cluster_id):
        """
        Extracts circles from a grayscale image based on a target cluster.

        Parameters:
        - gray_image: Grayscale input image.
        - target_cluster_id: Target cluster ID.

        Returns:
        - circles_info: List of tuples containing circle information (center_x, center_y, radius).
        - contours_: List of contours.
        """
        # Create a binary mask for the specified cluster
        cluster_mask = (gray_image == target_cluster_id).astype(np.uint8)

        cluster_mask2 = (cluster_mask * 255).astype('uint8')
        robert_edge = ImageProcessor.roberts_edge(cluster_mask2)
        robert_edge = (robert_edge * 255).astype('uint8')

        # Find contours in the binary mask
        contours, _ = cv2.findContours(robert_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get image dimensions
        height, width = gray_image.shape
        image_area = height * width

        # Initialize a list to store information about circles
        circles_info = []

        contours_ = []
        # Loop through contours and fit circles
        for contour in contours:
            # Fit a circle to the contour
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            tupp = (int(center_x), int(center_y), int(radius))

            # Calculate area of the circle
            circle_area = np.pi * (tupp[2] ** 2)

            # Check if the area is within the specified range (0.1 to 0.12 of image area)
            if 0.00005 * image_area <= circle_area <= 0.12 * image_area:
                # Append information about the circle to the list
                circles_info.append(tupp)

                contours_.append(contour)

        return circles_info, contours_

    def calculate_iou(self, circle1, circle2, image_shape):
        """
        Calculates the Intersection over Union (IoU) between two circles.

        Parameters:
        - circle1, circle2 (tuple): Circle parameters (center_x, center_y, radius).
        - image_shape (tuple): Shape of the image.

        Returns:
        - ratio: IoU ratio.
        """
        # check if circles are intersecting
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        y2 = image_shape[0] - y2
        d = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
        if r1 + r2 < d:
            return 0
        else:
            # Create masks for each circle
            mask1 = self.create_circle_mask((x1, y1), r1, image_shape)
            mask2 = self.create_circle_mask((x2, y2), r2, image_shape)

            # Calculate the intersection and union masks
            intersection = cv2.bitwise_and(mask1, mask2)

            # Count the number of pixels in the intersection and union
            intersection_count = np.sum(intersection)
            union_count = np.sum(mask2)

            # Calculate the ratio, return 0 if the circles are not touching
            ratio = intersection_count / union_count

            return ratio

    # find max iou
    def find_max_iou_circle(self, circles, expected_circle, image_shape, contours_):
        """
        Finds the circle with the maximum IoU compared to the expected circle.

        Parameters:
        - circles: List of tuples containing circle information (center_x, center_y, radius).
        - expected_circle: Circle parameters of the expected circle.
        - image_shape (tuple): Shape of the image.
        - contours_: List of contours.

        Returns:
        - max_iou_circle: Circle parameters of the circle with maximum IoU.
        - max_iou: Maximum IoU value.
        - contourr: Contour corresponding to the maximum IoU circle.
        """
        max_iou = 0
        max_iou_circle = None
        contourr = None

        i = 0
        for circle in circles:

            current_iou = self.calculate_iou(circle, expected_circle, image_shape)
            if current_iou > max_iou:
                max_iou = current_iou
                max_iou_circle = circle
                contourr = contours_[i]
            i = i + 1

        return max_iou_circle, max_iou, contourr

    def get_roi(self, df, window_size, is_norm=False):
        """
        Retrieves Region of Interest (ROI) from the given DataFrame.

        Parameters:
        - df: DataFrame containing image information.
        - window_size (int): Size of the windows.
        - is_norm (bool): Whether to process normal images.

        Returns:
        - rois: List of extracted ROIs.
        """
        rois = list()
        if is_norm:
            inds = df.loc[df['abnormality_type'] == 'NORM'].index.tolist()
        else:
            inds = df.loc[df['abnormality_type'] != 'NORM'].index.tolist()

        for ind in tqdm(inds):
            if not is_norm:
                rf, x, y, rr = df.loc[ind][['reference_number', 'ab_x', 'ab_y', 'ab_radius']]
                path = "data/images_aug/" + rf + ".pgm"
                img = self.img_processor.preprocess_image(path)
                r, h = img.shape

                x, y, rr = np.round([x, y, rr]).astype(int)
                y = h - y
                r = self.closest_multiple(rr, window_size)
                roi = img[y - r:y + r, x - r:x + r]
                rois.append(roi)
            else:
                rf = df.loc[ind]['reference_number']
                path = "data/images_aug/" + rf + ".pgm"
                img = self.img_processor.preprocess_image(path)
                h, w = img.shape
                x = w // 2
                y = h - h // 3
                rr = min(max(50, np.round(0.16 * x).astype('int32')), 124)
                r = self.closest_multiple(rr, window_size)
                x, y, r = np.round([x, y, r]).astype(int)

            roi = img[y - r:y + r, x - r:x + r]
            rois.append(roi)

        return rois
