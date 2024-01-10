import cv2
import string
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Utilities:
    @staticmethod
    def read_image(path, img_size=400, no_resize=False):
        img_full = mpimg.imread(path)
        if not no_resize:
            img_full = cv2.resize(img_full, (img_size, img_size))
        return img_full

    @staticmethod
    def right_orient_mammogram(image):
        left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1] / 2)])
        right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2):])

        if left_nonzero < right_nonzero:
            image = cv2.flip(image, 1)

        return image

    @staticmethod
    def get_orientation(image):
        left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1] / 2)])
        right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2):])

        if left_nonzero < right_nonzero:
            return 'right'

        return 'left'

    @staticmethod
    def plot_array(r, c, img_ar):
        ind = 0
        length_img = len(img_ar)
        alphabets = string.ascii_lowercase

        if r > 1:
            fig, axes = plt.subplots(r, c, figsize=(12, 8))
            for i in range(r):
                for j in range(c):
                    img = img_ar[ind]
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f"Figure 1{alphabets[ind]}", pad=10)
                    ind += 1
                    if ind == length_img:
                        break
        else:
            fig, axes = plt.subplots(r, c, figsize=(12, 8))
            for j in range(c):
                img = img_ar[ind]
                axes[j].imshow(img, cmap='gray')
                axes[j].axis('off')
                ind += 1
                if ind == length_img:
                    break

        plt.show()

# Example usage:
# image = Utilities.read_image('path/to/your/image.jpg')
# processed_image = Utilities.right_orient_mammogram(image)
# orientation = Utilities.get_orientation(image)
# Utilities.plot_array(2, 2, [image, processed_image, image, processed_image])
