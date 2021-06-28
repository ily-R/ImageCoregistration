"""
Created on Thu Dec 06 10:24:14 2019

@author: ilyas Aroui
"""

import cv2
import numpy as np
import os
from sklearn import linear_model
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def display_matches(img1, img2, kp1, kp2, name, num=20, save=False):
    """Helper to display matches of keypoint in botch images, by connecting a line from one image to another

    Typical use:
        display_matches(target, source, lmk1, lmk2, name="matches", save = True)

    img1, img2: target and source images as np.ndarray
    kp1, kp2: landmarks of target and source images respectively as np.ndarray
    name: name of the figure display and the image saved if save = True
    save: boolean indicates to save the image of the matches
    """
    if img1.shape[0] != img2.shape[0]:
        minn = min(img1.shape[0], img1.shape[0])
        if minn == img1.shape[0]:
            img1 = np.concatenate(
                (img1, np.zeros(img2.shape[0] - minn, img1.shape[1], 3)), axis=0
            )
        else:
            img2 = np.concatenate(
                (img2, np.zeros(img1.shape[0] - minn, img2.shape[1], 3)), axis=0
            )
    img = np.concatenate((img1, img2), axis=1)
    for i in np.random.choice(len(kp1), min(num, len(kp1))):
        x1, y1 = int(kp1[i][0]), int(kp1[i][1])
        x2, y2 = int(kp2[i][0]) + img1.shape[1], int(kp2[i][1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, img.shape[1], img.shape[0])
    cv2.imshow(name, img)
    cv2.imwrite(os.path.join("result", name + ".jpg"), img)


def match(lmk1, lmk2, desc1, desc2, sift_error=0.7):
    """Helper to find the pair of matches between two keypoints lists
    it return two np.ndarray of landmarks in an order respecting the matching

    Typical use:
        lmk1, lmk2 = match(lmk1, lmk2, desc1, desc2)

    lmk1, lmk2: landmarks of target and source images respectively as np.ndarray
    desc1, desc2: descriptors of target and source images respectively as np.ndarray
    sift_error: if the ratio between the distance to the closest match and the second closest is less than sift_error
    reject this landmark.
    """
    match1, match2 = [], []
    for i in range(len(desc1)):
        distance = np.sqrt(np.sum((desc1[i] - desc2) ** 2, axis=1))
        indices = np.argsort(distance)
        if distance[indices[0]] / distance[indices[1]] < sift_error:
            match1.append(lmk1[i])
            match2.append(lmk2[indices[0]])
    return np.array(match1), np.array(match2)


def cross_corr(img1, img2):
    """Helper to calculate cross_correlation metric between two images. Well adapted, if we assume there is a linear
    transformation between pixels intensities in both images.
    it returns the cross-correlation value.

    Typical use:
        cc = cross_corr(warped, target_w)
    """
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mean1, mean2 = np.mean(img1), np.mean(img2)
    img1, img2 = img1 - mean1, img2 - mean2
    numerator = np.sum(np.multiply(img1, img2))
    denominator = np.sqrt(
        np.sum(np.multiply(img1, img1)) * np.sum(np.multiply(img2, img2))
    )
    corr = numerator / denominator
    print("Cross-correlation: ", corr)
    return corr


def mutual_inf(img1, img2, verbose=False):
    """Helper to calculate mutual-information metric between two images. it gives a probabilistic measure on how
    uncertain we are about the target image in the absence/presence of the warped source image
    it returns the mutual information value.

    Typical use:
        mi = mutual_inf(warped, target_w)

    verbose: if verbose=True, display and save the joint-histogram between the two images.
    """
    epsilon = 1.0e-6
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = np.round(img1).astype("uint8")
    img2 = np.round(img2).astype("uint8")

    joint_hist = np.zeros((256, 256))
    for i in range(min(img1.shape[0], img2.shape[0])):
        for j in range(min(img1.shape[1], img2.shape[1])):
            joint_hist[img1[i, j], img2[i, j]] += 1

    if verbose:
        display_jh = np.log(joint_hist + epsilon)
        display_jh = (
            255
            * (display_jh - display_jh.min())
            / (display_jh.max() - display_jh.min())
        )
        cv2.imshow("joint_histogram", display_jh)
        cv2.imwrite("result/joint_histogram.jpg", display_jh)

    joint_hist /= np.sum(joint_hist)
    p1 = np.sum(joint_hist, axis=0)
    p2 = np.sum(joint_hist, axis=1)
    joint_hist_d = joint_hist / (p1 + epsilon)
    joint_hist_d /= p2 + epsilon
    mi = np.sum(np.multiply(joint_hist, np.log(joint_hist_d + epsilon)))
    print("Mutual Information: ", mi)
    return mi


def ransac(kp1, kp2):
    """Helper to apply ransac (RANdom SAmple Consensus) algorithm on two arrays of landmarks
    it returns the inliers and outliers in both arrays

    Typical use:
        lmk1, lmk2, outliers1, outliers2 = ransac(lmk1, lmk2)

    kp1, kp2: landmarks of target and source images respectively as np.ndarray
    """
    ransac_model = linear_model.RANSACRegressor()
    ransac_model.fit(kp1, kp2)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    return kp1[inlier_mask], kp2[inlier_mask], kp1[outlier_mask], kp2[outlier_mask]


def calculate_transform(kp1, kp2):
    """Helper to apply find the best affine transform using two arrays of landmarks.
    it returns the affine transform, a matrix T of size (2, 3)

    Typical use:
        T = calculate_transform(lmk2, lmk1)

    kp1, kp2: landmarks of target and source images respectively as np.ndarray
    """
    upper = np.concatenate(
        (kp1, np.ones((kp1.shape[0], 1)), np.zeros((kp1.shape[0], 3))), axis=1
    )
    lower = np.concatenate(
        (np.zeros((kp1.shape[0], 3)), kp1, np.ones((kp1.shape[0], 1))), axis=1
    )
    X = np.concatenate((upper, lower), axis=0)
    Y = np.concatenate((kp2[:, 0], kp2[:, 1]))
    Y = np.expand_dims(Y, axis=-1)
    T = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    T = T.reshape(2, 3)
    T = np.concatenate((T, np.array([0, 0, 1]).reshape(1, 3)), axis=0)
    kp2_pred = np.dot(T, np.concatenate((kp1, np.ones((kp1.shape[0], 1))), axis=1).T).T
    kp2_pred /= kp2_pred[:, -1:]
    error = np.linalg.norm(kp2_pred[:, :2] - kp2)
    print("coordinate reconstruction error: ", error)
    return T


def warp(target, source, T):

    """Helper to move the source image to the same reference as target image, so they can be co-registered.
    it returns the new warped source image and the target image which is also centered in a larger figure by 10 pixels.
    i.e, if the input size is (M, N) then the output is (M+10, N+10).

    Typical use:
        warped, target_w = warp(target, source, T)

    T:  affine transform, a matrix T of size (2, 3)
    """
    height = target.shape[0]
    width = source.shape[1]

    # move both images to the center a bit
    corners = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    corners_moved = np.float32(
        [[5, 5], [5, height + 5], [5 + width, 5], [5 + width, 5 + height]]
    )
    T_perspective = cv2.getPerspectiveTransform(corners, corners_moved)
    target_new = cv2.warpPerspective(target, T_perspective, (width + 10, height + 10))
    cv2.imshow("target_new", target_new)
    cv2.imwrite("result/target_new.jpg", target_new)
    T = np.dot(T_perspective, T)
    source_new = cv2.warpPerspective(
        source, T, (width + 10, height + 10), cv2.INTER_AREA
    )
    cv2.imshow("source_new", source_new)  # show transform
    cv2.imwrite("result/source_new.jpg", source_new)
    return source_new, target_new
