# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:06:59 2018

@author: Bllue
"""

import cv2 
import numpy as np

def angle_rad(v1, v2):
    """Angle in radians between two vectors

        This method returns the angle (in radians) between two array-like
        vectors using the cross-product method, which is more accurate for
        small angles than the dot-product-acos method.
    """
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


def deg2rad(angle_deg):
    """Convert degrees to radians

        This method converts an angle in radians e[0,2*np.pi) into degrees
        e[0,360)
    """
    return angle_deg/180.0*np.pi


height = 480
width = 640

img = cv2.imread('hand.png')
img = cv2.resize(img, (640, 480))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.circle(img, (int(width/2), int(height/2)), 3, [255, 102, 0], 2)
cv2.rectangle(img, (int(width/3), int(height/3)), (int(width*2/3), int(height*2/3)),
              [255, 102, 0], 2)
        
cv2.imshow('guisture', img)
#cv2.waitKey(0)


small_kernel = 3
#height = 480
#width = 640

center_half = 10  # half-width of 21 is 21/2-1
center = img[int(height/2-center_half) : int(height/2+center_half),
               int(width/2-center_half) : int(width/2+center_half)]

# find median depth value of center region
med_val = np.median(center)

# try this instead:
abs_depth_dev = 14
img = np.where(abs(img-med_val) <= abs_depth_dev,
                 128, 0).astype(np.uint8)
        
kernel = np.ones((3,3), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('morph close', img)
#cv2.waitKey(0)


img[int(height/2-small_kernel) : int(height/2+small_kernel), 
    int(width/2-small_kernel) : int(width/2+small_kernel)] = 128
#cv2.imshow('a', img)
#cv2.waitKey(0)


mask = np.zeros((height+2, width+2), np.uint8)

flood = img.copy()

cv2.floodFill(flood, mask, (int(width/2), int(height/2)), 255, flags=4|(255<<8))
cv2.imshow('flood', flood)

ret, flooded = cv2.threshold(flood, 129, 255, cv2.THRESH_BINARY)


cv2.imshow('flooded', flooded)

img, contours, hierarchy = cv2.findContours(flooded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


max_contour = max(contours, key=cv2.contourArea)
epsilon = 0.01*cv2.arcLength(max_contour, True)
max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

# find convexity hull and defects
hull = cv2.convexHull(max_contour, returnPoints=False)
defects = cv2.convexityDefects(max_contour, hull)

#cv2.imshow('a', img)

contours = max_contour
img_draw = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
if defects is None:
    print(0)

# we assume the wrist will generate two convexity defects (one on each
# side), so if there are no additional defect points, there are no
# fingers extended
if len(defects) <= 2:
    print(0)

# if there is a sufficient amount of convexity defects, we will find a
# defect point between two fingers so to get the number of fingers,
# start counting at 1
num_fingers = 1

for i in range(defects.shape[0]):
    # each defect point is a 4-tuple
    start_idx, end_idx, farthest_idx, _ = defects[i, 0]
    start = tuple(contours[start_idx][0])
    end = tuple(contours[end_idx][0])
    far = tuple(contours[farthest_idx][0])

    # draw the hull
    cv2.line(img_draw, start, end, [0, 255, 0], 2)

    # if angle is below a threshold, defect point belongs to two
    # extended fingers
    thresh_deg = 80.0
    if angle_rad(np.subtract(start, far),
                 np.subtract(end, far)) < deg2rad(thresh_deg):
        # increment number of fingers
        num_fingers = num_fingers + 1

        # draw point as green
        cv2.circle(img_draw, far, 5, [0, 255, 0], -1)
    else:
        # draw point as red
        cv2.circle(img_draw, far, 5, [255, 0, 0], -1)
print(min(5, num_fingers))
cv2.imshow('result', img_draw)
cv2.waitKey(0)
    



