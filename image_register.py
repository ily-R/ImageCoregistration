"""
Created on Thu Dec 06 10:24:14 2019

@author: ilyas Aroui
"""
import cv2
import numpy as np
import argparse
import os
from utils import *


def readAndRescale(img1, img2, scale):
    """Helper to read images, scale them and convert to grayscale.
        it returns original, gray and scaled images

    Typical use:
        t, s, t_gray, s_gray, t_full, s_full = readAndRescale("cat1.jpg", "cat2.jpg", 0.3)

    img1: target image name
    img2: source image name
    scale: scaling factor, keeping aspect ratio
    """
    target = cv2.imread(os.path.join("data", img1))
    source = cv2.imread(os.path.join("data", img2))

    width = int(target.shape[1] * scale)
    height = int(source.shape[0] * scale)
    dim = (width, height)

    target_s = cv2.resize(target, dim, interpolation=cv2.INTER_AREA)
    source_s = cv2.resize(source, dim, interpolation=cv2.INTER_AREA)

    gray1 = cv2.cvtColor(target_s, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(source_s, cv2.COLOR_BGR2GRAY)

    return target_s, source_s, gray1, gray2, target, source


def getKeypointAndDescriptors(target_gray, source_gray):
    """Helper to get Harris points of interest and use them as landmarks for sift descriptors.
        it returns these landmarks and their descriptors

    Typical use:
        lmk1, lmk2, desc1, desc2 = getKeypointAndDescriptors(target_gray, source_gray)

    target_gray, source_gray: grayscaled target and source images as np.ndarray
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(target_gray, None)
    pts1 = np.array([kp1[idx].pt for idx in range(len(kp1))])
    kp2, des2 = sift.detectAndCompute(source_gray, None)
    pts2 = np.array([kp2[idx].pt for idx in range(len(kp2))])
    return pts1, pts2, des1, des2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image name")
    parser.add_argument("source", help="source image name")
    parser.add_argument('-s', '--scale', default=.1, type=float,
                        help='rescale the images with this factor in range [0, 1]')

    parser.add_argument("-r","--ransac", help="apply ransac", action="store_true")

    args = parser.parse_args()
    target, source, target_gray, source_gray, target_full, source_full = readAndRescale(args.target, args.source, args.scale)

    lmk1, lmk2, desc1, desc2 = getKeypointAndDescriptors(target_gray, source_gray)

    lmk1, lmk2 = match(lmk1, lmk2, desc1, desc2)
    display_matches(target, source, lmk1, lmk2, name="matches")
    if args.ransac:
        lmk1, lmk2, outliers1, outliers2 = ransac(lmk1, lmk2)
        display_matches(target, source, outliers1, outliers2, name="matches_removed_by_RANSAC", num=5, save=True)

    T = calculate_transform(lmk2, lmk1)
    warped, target_w = warp(target, source, T)
    cc = cross_corr(warped, target_w)
    mi = mutual_inf(warped, target_w, verbose=True)

