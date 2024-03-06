
# ======================================================================================================================

# Various helper functions

# Author: Jonas Anderegg, Simon Treier, Flavian Tschurr, ETH Zürich
# Last edited: 2024-03-05

# ======================================================================================================================

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
import geojson

np.seterr(divide='ignore', invalid='ignore')


# vegetation index
def calculate_index(img):
    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img)

    normalizer = np.array(R.astype("float32") + G.astype("float32") + B.astype("float32"))

    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    return ExG


def color_index_transformation(image):
    R, G, B = cv2.split(image)
    normalizer = np.array(R.astype("float32") + G.astype("float32") + B.astype("float32"))
    normalizer[normalizer == 0] = np.float32(1)  # avoid division by 0
    r, g, b = (R, G, B) / normalizer
    # Green Leaf Index
    GLI = np.array((2 * g - r - b) / (2 * g + r + b), dtype=np.float32)
    # Excess Green Index
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)
    # Excess Red Index
    ExR = np.array(1.3 * r - g)
    # Normalized Difference Index
    NDI = 128 * (((g - r) / (g + r)) + 1)
    # Excess Green minus Excess red Index
    ExGR = (2 * g - (r + b)) - (1.3 * r - g)
    # Normalized Green Red Difference Index
    NGRDI = np.array((g - r) / (g + r), dtype=np.float32)
    # Triangular greenness index
    lambda_R = 670
    lambda_G = 550
    lambda_B = 480
    TGI = -0.5 * ((lambda_R - lambda_B) * (r - g) - (lambda_R - lambda_G) * (r - b))
    # Vegetation Index
    r[r == 0] = 0.00001  # Avoid division by zero
    b[b == 0] = 0.00001  # Avoid division by zero
    VEG = g / ((r ** 0.667) * (b ** 0.333))

    desc = [GLI, ExG, ExR, NDI, ExGR, NGRDI, TGI, VEG]
    desc_names = ['GLI', 'ExG', 'ExR', 'NDI', 'ExGR', 'NGRDI', 'TGI', 'VEG']

    return desc, desc_names


def index_distribution(image, feature_name, level_id, level_mask):
    px_roi = image[level_mask == 255]
    mn = np.nanmean(px_roi)
    md = np.nanmedian(px_roi)
    kt = kurtosis(px_roi, nan_policy='omit')
    sk = skew(px_roi, nan_policy='omit')
    p75, p251 = np.nanpercentile(px_roi, [75, 25])
    iqr = p75 - p251
    p98, p02 = np.nanpercentile(px_roi, [98, 2])
    ipr = p98 - p02
    std = np.nanstd(px_roi)
    stat_names = ["mean", "median", "kurtosis", "skewness", "intqrange", "intprange", "stddev"]
    stat_names = [level_id + "_" + feature_name + "_" + n for n in stat_names]
    return [mn, md, kt, sk, iqr, ipr, std], stat_names


def _check_geojson(geojson_file):
    """ check if a certain file name contains to a geojson file
    :param geojson_file: input filename
    :return: True/ False
    """
    format = ".geojson"
    return format in geojson_file


def get_homography_matrix(source, destination):
    """ Calculates the entries of the Homography matrix between two sets of matching points.
    Args
    ----
        - `source`: Source points where each point is int (x, y) format.
        - `destination`: Destination points where each point is int (x, y) format.
    Returns
    ----
        - A numpy array of shape (3, 3) representing the Homography matrix.
    Raises
    ----
        - `source` and `destination` is lew than four points.
        - `source` and `destination` is of different size.
    """
    assert len(source) >= 4, "must provide more than 4 source points"
    assert len(destination) >= 4, "must provide more than 4 destination points"
    assert len(source) == len(destination), "source and destination must be of equal length"
    A = []
    b = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))


def get_coords_from_geojson(path_json):

    with open(path_json, 'r') as infile:
        polygon_mask = geojson.load(infile)
    polygons = polygon_mask["features"]
    for polygon in polygons:
        if len(polygon["geometry"]["coordinates"][0]) == 5:
            coords = polygon["geometry"]["coordinates"][0]
        else:
            coords = polygon["geometry"]["coordinates"][0][0]

    return coords


def warp_point(x: int, y: int, M) -> [int, int]:
    """
    Applies a homography matrix to a point
    :param x: the x coordinate of the point
    :param y: the y coordinates of the point
    :param M: the homography matrix
    :return: coordinates of the warped point
    """
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    return ([
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    ])