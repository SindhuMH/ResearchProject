import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from image_preprocess import ImageProcessor
from extract_roi import RoiExtract
from models import Models


class MammogramSegmentor:

    def __init__(self):
        self.img_processor = ImageProcessor()
        self.roi_ext = RoiExtract()
        self.models = Models()

    @staticmethod
    def split_data_with_proportion(df, test_size=0.3, random_state=42):

        # Split the "df" rows into train and test
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    @staticmethod
    def get_results(iouscores):

        detection_percents = np.empty((6, 4), dtype='object')
        model_name = "K-means"
        tissue_type = ["Fatty", "Glandular", "Dense"]

        j = 0
        for i in range(6):
            iousi = iouscores[i]
            det = 0
            undet = 0
            tot = len(iousi)
            for iou in iousi.values():
                if iou > 0:
                    det = det + 1
                else:
                    undet = undet + 1
            detection_percents[i, :] = [tissue_type[j], model_name, (det / tot) * 100, (undet / tot) * 100]
            if i >= 2:
                model_name = "GMM"
            j = j + 1
            j = j % 3

        detection_df = pd.DataFrame(detection_percents,
                                    columns=["BreastTissue", "SegmentationModel", "Detection%", "Undetection%"])
        print(detection_df)

    def load_data(self, percentt=0.85):
        df = pd.read_csv('data/aug_df.csv')
        gdf = df.loc[df['bg_tissue'] == 'G'].reset_index(drop=True)
        fdf = df.loc[df['bg_tissue'] == 'F'].reset_index(drop=True)
        ddf = df.loc[df['bg_tissue'] == 'D'].reset_index(drop=True)

        tr_f, ts_f = self.split_data_with_proportion(fdf, test_size=percentt)
        tr_g, ts_g = self.split_data_with_proportion(gdf, test_size=percentt)
        tr_d, ts_d = self.split_data_with_proportion(ddf, test_size=percentt)

        shapes = np.transpose([[tr_f.shape[0], tr_g.shape[0], tr_d.shape[0]],
                               [ts_f.shape[0], ts_g.shape[0], ts_d.shape[0]]])
        shapes_df = pd.DataFrame(shapes, columns=['train_data_size', 'test_data_size'],
                                 index=['Fatty tissue', 'Glandular Tissue', 'Dense Tissue'])
        print(shapes_df)

        return tr_f, ts_f, tr_g, ts_g, tr_d, ts_d

    def train_models(self, dtr, dts, bg, evaluate=False):
        num_clusters = 4
        window_size = 2

        kmeans_f, target_km = self.models.get_kmeans(dtr, num_clusters, window_size)

        gmm_f = self.models.get_gmm(dtr, window_size, kmeans_model=kmeans_f, num_components=num_clusters)

        inds = dts.loc[dts['abnormality_type'] != 'NORM'].index.tolist()
        ind = inds[11]
        print(dts.loc[ind])

        target_gm = float(target_km)
        okay = False
        iou = self.models.get_preds(ind, dts, target_gm, gmm_f, window_size=window_size, draw=True)
        print("iou", iou)

        while ~okay:
            target_gm = input(f"please choose cluster label for abnormality between 1 to {num_clusters}")
            target_gm = float(target_gm)
            iou = self.models.get_preds(ind, dts, target_gm, gmm_f, window_size=window_size, draw=True)
            print("iou", iou)
            is_okay = input(f"is this okay Y or N")
            is_okay = is_okay.strip()
            if is_okay == 'Y':
                okay = True
                break

        inds = dts.loc[dts['abnormality_type'] != 'NORM'].index.tolist()
        iou_scores_f = dict()

        if not evaluate:
            return gmm_f, kmeans_f, None, None

        if evaluate:

            for ind in tqdm(inds):
                iou = self.models.get_preds(ind, dts, target_gm, gmm_f, window_size=window_size, draw=False)
                iou_scores_f[ind] = iou

            detected_f = []
            undetected_f = []

            for ind in iou_scores_f:
                score = iou_scores_f[ind]
                if score > 0:
                    detected_f.append(score)
                else:
                    undetected_f.append(ind)
            print("proportion of mammograms of type", bg, "for which mass was detected",
                  len(detected_f) / len(iou_scores_f))

            # Specify colors for each bin
            colors = {'D': 'lightorange', 'G': 'lightyellow', 'F': 'darkseagreen'}

            # Create the histogram with specified colors
            counts, bins, patches = plt.hist(list(iou_scores_f.values()), color=colors[bg], bins=[0, 0.01, 0.2, 1],
                                             edgecolor='black')

            # Add xlabel and ylabel
            plt.xlabel('Proportion of Overlap')
            plt.ylabel('Number of Images')
            plt.title(f'Distribution of ROI Overlap For {bg} Tissues')

            # Annotate each bar with its frequency
            for i in range(len(patches)):
                bar = patches[i]
                height = bar.get_height()
                plt.annotate(f'{height}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

            # Show the plot
            plt.show()

            kiou_scores_f = self.models.get_kmeans_ious(dts, window_size, kmeans_f, target_km)

            return gmm_f, kmeans_f, iou_scores_f, kiou_scores_f


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mseg = MammogramSegmentor()

    tr_f, ts_f, tr_g, ts_g, tr_d, ts_d = mseg.load_data()

    gmm_f, kmeans_f, iou_scores_f, kiou_scores_f = mseg.train_models(tr_f, ts_f, "F")

    gmm_g, kmeans_g, iou_scores_g, kiou_scores_g = mseg.train_models(tr_g, ts_g, "G")

    gmm_d, kmeans_d, iou_scores_d, kiou_scores_d = mseg.train_models(tr_d, ts_d, "D")

    iouscores = [kiou_scores_f, kiou_scores_g, kiou_scores_d, iou_scores_f, iou_scores_g, iou_scores_d]

    MammogramSegmentor.get_results(iouscores)
# /Users/sindhu/PycharmProjects/MastersResearchProject
