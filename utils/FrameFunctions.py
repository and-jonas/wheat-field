
# ======================================================================================================================

# Helper functions to find images containing a defined real-world position

# Authors: Flavian Tschurr, Jonas Anderegg
# Last modified: 2024-03-05

# ======================================================================================================================

import numpy as np
from scipy.spatial.distance import cdist


def image_finder(cornersDF, polygon_coords):

    poly_x = []
    poly_y = []
    for i in range(0, len(polygon_coords)):
        poly_x.append(polygon_coords[i][0])
        poly_y.append(polygon_coords[i][1])

    # frame centre
    mean_x = np.mean(poly_x)
    mean_y = np.mean(poly_y)

    dists = []
    for image in range(0, len(cornersDF.camera)):

        # get centre of image
        img_x = [float(cornersDF.loc[image:image].e1_x), float(cornersDF.loc[image:image].e2_x),
                 float(cornersDF.loc[image:image].e3_x), float(cornersDF.loc[image:image].e4_x)]
        img_y = [float(cornersDF.loc[image:image].e1_y), float(cornersDF.loc[image:image].e2_y),
                 float(cornersDF.loc[image:image].e3_y), float(cornersDF.loc[image:image].e4_y)]
        img_x = np.mean(img_x)
        img_y = np.mean(img_y)

        # calculate distance to centre of frame

        dist = cdist(np.asarray([[mean_x, mean_y]]), np.asarray([[img_x, img_y]]))[0][0]
        dists.append(dist)

    # find bests
    min_idx = dists.index(min(dists))
    identified_img = cornersDF.iloc[min_idx, 0]

    return identified_img

