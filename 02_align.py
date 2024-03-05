
# ======================================================================================================================

# Aligns the handheld and the UAV images and masks

# Author: Jonas Anderegg, ETH Zürich
# Last edited: 2024-03-05

# ======================================================================================================================

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import geojson
from PIL import Image
import cv2
import skimage
import imageio

from utils import base as utils
from utils import metrics as metrics
from matplotlib import path

# ======================================================================================================================
# Handheld on UAV
# ======================================================================================================================

fh = "10m"  # flight height
base = "O:/Evaluation/Hiwi/2023_herbifly_LTS"

# input dirs
coord_dir_hh = f"{base}/validation/frame_coord_handheld"
coord_dir_10m = f"{base}/validation/frame_coord_{fh}"
mask_dir_hh = f"{base}/validation/handheld/Output_vegann/SegVeg/Mask"
mask_dir_10m = f"{base}/validation/{fh}/masks_frames"
img_dir_10m = f"{base}/validation/{fh}/images_frames"

# output dirs
composite_dir = Path(f"{base}/validation/{fh}/Output/Composites")
warped_mask_dir = Path(f"{base}/validation/{fh}/Output/WarpedMasks")
warped_corners_dir = Path(f"{base}/validation/{fh}/Output/WarpedCorners")
data_dir = Path(f"{base}/validation/{fh}/Output/Data")
for d in [composite_dir, warped_mask_dir, warped_corners_dir, data_dir]:
    d.mkdir(exist_ok=True, parents=True)

# fields
# fields = ["Nennigkofen1", "Nennigkofen2", "Opfertshofen", "Volken", "Grafenried", "Steinmaur", "Treytorrens-P",
#           "Oleyres", "Villars-le-G"]
fields = ["Treytorrens-P",
          "Oleyres", "Villars-le-G"]

for field in fields:
    series = glob.glob(f"{base}/raw/{field}/Handheld/F[0-9]/*.JPG")
    id = [s[-6:-4] for s in series]
    uid = np.unique(id)
    for s in uid:

        # get and sort the series
        serie = glob.glob(f"{base}/raw/{field}/Handheld/F[0-9]/*{s}.JPG")
        dates = [os.path.basename(x).split("_")[0] for x in serie]
        sorter = np.argsort(dates)
        serie = [serie[index] for index in sorter]

        # process the series
        for idx, image in enumerate(serie):

            # file names
            bn = os.path.basename(image)
            sn = bn.replace(".JPG", "")
            on = "_".join(sn.split("_")[-3:])

            print(sn)

            # get images and masks
            img_hh = Image.open(image)
            img_hh = np.array(img_hh)
            img_name_10m = "_".join([on.split("_")[i] for i in [0, 1, 2]]).replace("_F", "_Frame") + ".JPG"
            img_name_10m_base = img_name_10m.replace(".JPG", "")

            try:
                img_10m = Image.open(f'{img_dir_10m}/{img_name_10m}')
                img_10m = np.array(img_10m)
            except:
                try:
                    # day before
                    date_corr = int(img_name_10m.split("_")[0]) - 1
                    img_name_10m = "_".join([str(date_corr)] + img_name_10m.split("_")[1:])
                    img_10m = Image.open(f'{img_dir_10m}/{img_name_10m}')
                    img_10m = np.array(img_10m)
                except:
                    try:
                        # day after
                        date_corr = int(img_name_10m.split("_")[0]) + 1
                        img_name_10m = "_".join([str(date_corr)] + img_name_10m.split("_")[1:])
                        img_10m = Image.open(f'{img_dir_10m}/{img_name_10m}')
                        img_10m = np.array(img_10m)
                    except:
                        print("Error")
                        continue

            mask_hh = Image.open(f'{mask_dir_hh}/{bn.replace(".JPG", ".png")}')
            mask_hh = np.array(mask_hh)
            mask_10m = Image.open(f'{mask_dir_10m}/{img_name_10m.replace(".JPG", ".png")}')
            mask_10m = np.array(mask_10m)

            path_geojson_hh = f'{coord_dir_hh}/{sn}.geojson'
            coords_hh = utils.get_coords_from_geojson(path_geojson_hh)

            path_geojson_10m = f'{coord_dir_10m}/{img_name_10m.replace(".JPG", ".geojson")}'
            coords_10m = utils.get_coords_from_geojson(path_geojson_10m)

            h = utils.get_homography_matrix(coords_hh, coords_10m)
            warped_img = cv2.warpPerspective(img_hh, h, (img_10m.shape[1], img_10m.shape[0]))
            warped_img = skimage.util.img_as_ubyte(warped_img)
            warped_mask = cv2.warpPerspective(mask_hh, h, (img_10m.shape[1], img_10m.shape[0]))
            warped_mask = skimage.util.img_as_ubyte(warped_mask)
            warped_mask = np.where(warped_mask > 100, 255, 0)

            # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            # axs[0].imshow(img_10m)
            # axs[0].set_title('leaf')
            # axs[1].imshow(mask_10m)
            # axs[1].set_title('warped')
            # axs[2].imshow(warped_mask)
            # axs[2].set_title('warped')
            # plt.show(block=True)

            # transform coordinates to a path
            grid_path = path.Path(coords_10m, closed=False)

            # create a mask of the image
            xcoords = np.arange(0, img_10m.shape[0])
            ycoords = np.arange(0, img_10m.shape[1])
            coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

            # Create mask
            ref_mask = grid_path.contains_points(coords, radius=-0.5)
            ref_mask = np.swapaxes(ref_mask.reshape(img_10m.shape[1], img_10m.shape[0]), 0, 1)
            ref_mask_ = np.where(ref_mask, True, False)
            ref_mask_ = np.stack([ref_mask_, ref_mask_, ref_mask_], axis=2)

            composite = np.where(ref_mask_, warped_img, img_10m)

            # metrics
            # transform the corner coordinates of the handheld image
            corners = [[0, 0], [0, mask_hh.shape[0]], [mask_hh.shape[1], mask_hh.shape[0]], [mask_hh.shape[1], 0], [0, 0]]
            corners_warped = [utils.warp_point(x[0], x[1], h) for x in corners]

            # corners_warped = np.intp(cv2.transform(np.array([corners]), h))[0].astype("float")
            # corners_warped = utils.make_point_list_(corners_warped)
            # transform coordinates to a path
            grid_path = path.Path(corners_warped, closed=False)

            # create a mask of the image
            xcoords = np.arange(0, img_10m.shape[0])
            ycoords = np.arange(0, img_10m.shape[1])
            coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

            # Create mask
            ref_mask = grid_path.contains_points(coords, radius=-0.5)
            ref_mask = np.swapaxes(ref_mask.reshape(img_10m.shape[1], img_10m.shape[0]), 0, 1)
            ref_mask_ = np.where(ref_mask, True, False)
            ref_mask_3d = np.stack([ref_mask_, ref_mask_, ref_mask_], axis=2)

            # composite = np.where(ref_mask_3d, warped_img, img_10m)
            #
            # fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
            # axs[0].imshow(img_10m)
            # axs[0].set_title('leaf')
            # axs[1].imshow(mask_10m)
            # axs[1].set_title('warped')
            # axs[2].imshow(warped_mask)
            # axs[2].set_title('warped')
            # axs[3].imshow(composite)
            # axs[3].set_title('warped')
            # plt.show(block=True)
            #
            # mask_10m_figure = np.stack([mask_10m, mask_10m, mask_10m], axis=2)
            # cv2.drawContours(mask_10m_figure, [grid_path.vertices[:5].astype(int)], 0, (255, 0, 0), 9)
            #
            # for j, k in enumerate([img_10m, mask_10m_figure, warped_mask, composite]):
            #     roi_crop = k[1500:2600, 2400:3200]
            #     roi_crop = roi_crop.astype("uint8")
            #     imageio.imwrite(f"{root}/Evaluation/Hiwi/2020_Herbifly/Publications/2023_Herbifly/Figures/multi/img_{j}.png", roi_crop)

            pix_tot = np.sum(ref_mask)

            warped_mask_bool = np.where(warped_mask == 255, True, False)
            mask_10m_bool = np.where(mask_10m == 255, True, False)

            mask_roi_hh = np.where(ref_mask_, warped_mask_bool, np.nan)
            mask_roi_uav = np.where(ref_mask_, mask_10m_bool, np.nan)

            # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            # axs[0].imshow(img_10m)
            # axs[0].set_title('leaf')
            # axs[1].imshow(mask_roi_hh)
            # axs[1].set_title('warped')
            # axs[2].imshow(mask_roi_uav)
            # axs[2].set_title('warped')
            # plt.show(block=True)

            mets = metrics.calculate_metrics(mask1=mask_roi_uav, mask2=mask_roi_hh)
            iou, dice, precision, recall, accuracy = mets

            cc_hh = np.nansum(mask_roi_hh)/pix_tot
            cc_uav = np.nansum(mask_roi_uav)/pix_tot

            # save outputs
            imageio.imwrite(composite_dir / img_name_10m, composite)
            imageio.imwrite(warped_mask_dir / img_name_10m.replace(".JPG", ".png"), warped_mask.astype("uint8"))

            polygon = geojson.Polygon([corners_warped])
            feature = geojson.Feature(geometry=polygon,
                                      properties={'image': img_name_10m_base, 'type': 'handheld_mask'})
            feature_collection = geojson.FeatureCollection(features=[feature])
            with open(f'{warped_corners_dir}/{img_name_10m_base}.geojson', 'w') as outfile:
                geojson.dump(feature_collection, outfile, indent='\t')

            frame_data = []
            frame_data.append({'label': img_name_10m,
                                'precision': precision,
                                'recall': recall,
                                'accuracy': accuracy,
                                'iou': iou,
                                'dice': dice,
                                'cc_hh': cc_hh,
                                'cc_uav': cc_uav})
            df = pd.DataFrame(frame_data)
            df.to_csv(f'{data_dir / img_name_10m.replace(".JPG", ".csv")}', index=False)

