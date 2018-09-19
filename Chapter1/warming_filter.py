# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:08:11 2018

@author: Bllue
"""

import cv2
from scipy.interpolate import UnivariateSpline
import numpy as np


def create_LUT_8UC1( x, y):
    """Creates a look-up table using scipy's spline interpolation"""
    spl = UnivariateSpline(x, y)
    return spl(range(256))
    
    
img_rgb = cv2.imread('me.jpg')
img_rgb = cv2.resize(img_rgb, (540, 720))
cv2.imshow('origin',img_rgb)


incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                              [0, 70, 140, 210, 256])
decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                              [0, 30,  80, 120, 192])
c_r, c_g, c_b = cv2.split(img_rgb)
c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
img_rgb = cv2.merge((c_r, c_g, c_b))

c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)

img_warming = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

cv2.imshow('warming filter',img_warming)
cv2.waitKey(0)