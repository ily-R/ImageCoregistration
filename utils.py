"""
Created on Thu Dec 06 10:24:14 2019

@author: ilyas Aroui
"""

import cv2
import numpy as np


def display_matches(img1, img2, kp1, kp2, num=20, save=None):
    if img1.shape[0] != img2.shape[0]:
        minn = min(img1.shape[0], img1.shape[0])
        if minn == img1.shape[0]:
            img1 = np.concatenate((img1, np.zeros(img2.shape[0] - minn, img1.shape[1], 3)), axis=0)
        else:
            img2 = np.concatenate((img2, np.zeros(img1.shape[0] - minn, img2.shape[1], 3)), axis=0)
    img = np.concatenate((img1, img2), axis=1)
    for i in np.random.choice(len(kp1), num):
        x1, y1 = int(kp1[i][0]), int(kp1[i][1])
        x2, y2 = int(kp2[i][0]) + img1.shape[1], int(kp2[i][1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.namedWindow('jpg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('jpg', img.shape[1], img.shape[0])
    cv2.imshow('jpg', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    if save is not None:
        cv2.imwrite(save, img)


def match(lmk1, lmk2, desc1, desc2, sift_error=0.7):
    match1, match2 = [], []
    for i in range(len(desc1)):
        distance = np.sqrt(np.sum((desc1[i] - desc2) ** 2, axis=1))
        indices = np.argsort(distance)
        if distance[indices[0]] / distance[indices[1]] < sift_error:
            match1.append(lmk1[i])
            match2.append(lmk2[indices[0]])
    return np.array(match1), np.array(match2)


def cross_corr(img1, img2):
    pass


def mutual_inf(img1, img2):
    pass


def ransac(kp1, kp2):
    pass


def calculate_transform(kp1, kp2):
    upper = np.concatenate((kp1, np.ones((kp1.shape[0], 1)), np.zeros((kp1.shape[0], 3))), axis=1)
    lower = np.concatenate((np.zeros((kp1.shape[0], 3)), kp1, np.ones((kp1.shape[0], 1))), axis=1)
    X = np.concatenate((upper, lower), axis=0)
    Y = np.concatenate((kp2[:, 0], kp2[:, 1]))
    Y = np.expand_dims(Y, axis=-1)
    T = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    T = T.reshape(2, 3)
    T = np.concatenate((T, np.array([0, 0, 1]).reshape(1, 3)), axis=0)
    kp2_pred = np.dot(T, np.concatenate((kp1, np.ones((kp1.shape[0], 1))), axis=1).T).T
    kp2_pred /= kp2_pred[:, -1:]
    error = np.linalg.norm(kp2_pred[:, :2] - kp2)
    print("reconstruction error: ", error)
    return T


def warp(target, source, T):
    height = target.shape[0]
    width = source.shape[1]

    # move both images to the center a bit
    corners = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    corners_moved = np.float32([[5, 5], [5, height + 5], [5 + width, 5], [5 + width, 5 + height]])
    T_perspective = cv2.getPerspectiveTransform(corners, corners_moved)
    target_new = cv2.warpPerspective(target, T_perspective, (width + 10, height + 10))
    cv2.imshow("target_new", target_new)
    cv2.imwrite("target_new.jpg", target_new)
    T = np.dot(T_perspective, T)
    source_new = cv2.warpPerspective(source, T, (width + 10, height + 10), cv2.INTER_AREA)
    cv2.imshow("source_new", source_new)  # show transform
    cv2.imwrite("source_new.jpg", source_new)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return source_new
