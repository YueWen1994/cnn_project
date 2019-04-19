import numpy as np
import cv2


def create_contour_msk(img, threshold=0.2):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)

    sobel_magtitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edges = sobel_magtitude > threshold * np.max(sobel_magtitude)

    # Perform some smoothing to eliminate the spots
    smoothed = np.uint8(edges.copy())
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, np.ones((3, 3)))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, np.ones((5, 5)))
    contours, _ = cv2.findContours(smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.asarray(contours)

    area = np.array([cv2.contourArea(contour) for contour in contours])
    bbox = np.array([cv2.boundingRect(contour) for contour in contours]).astype(float)
    aspect_ratio = bbox[:,2].flatten()/bbox[:,3].flatten()

    is_valid = (area > 200) & (aspect_ratio<3) & (aspect_ratio > 0.25)

    mask = np.ones(smoothed.shape[:2],dtype = 'uint8') * 255
    for ix in range(0,len(is_valid),1):
        if is_valid[ix] == False:
           cv2.drawContours(mask,[contours[ix]],-1,0,-1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((2, 2)))
    smoothed = cv2.bitwise_and(smoothed, mask)
    final_msk = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, np.ones((2, 2)))
    return final_msk