import numpy as np


def calculate_metrics(mask1, mask2):

    valid_pixels_mask1 = np.bitwise_not(np.isnan(mask1))
    valid_pixels_mask2 = np.bitwise_not(np.isnan(mask2))

    intersect = np.sum(mask2[valid_pixels_mask2]*mask1[valid_pixels_mask1])
    total_pixel_pred = np.sum(mask1[valid_pixels_mask1])
    precision = np.mean(intersect/total_pixel_pred)

    total_pixel_truth = np.sum(mask2[valid_pixels_mask2])
    recall = np.mean(intersect/total_pixel_truth)

    union = np.sum(mask1[valid_pixels_mask1]) + np.sum(mask2[valid_pixels_mask2]) - intersect
    xor = np.sum(mask2[valid_pixels_mask2] == mask1[valid_pixels_mask1])
    accuracy = np.mean(xor/(union + xor - intersect))

    total_sum = np.sum(mask1[valid_pixels_mask1]) + np.sum(mask2[valid_pixels_mask2])
    dice = np.mean(2*intersect/total_sum)

    iou = np.mean(intersect/union)

    return iou, dice, precision, recall, accuracy

