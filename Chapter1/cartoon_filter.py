# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:12:34 2018

@author: Bllue
"""
import cv2

img_rgb = cv2.imread('me.jpg')
img_rgb = cv2.resize(img_rgb, (540, 720))
cv2.imshow('origin',img_rgb)

numDownSamples = 2       # number of downscaling steps
numBilateralFilters = 7  # number of bilateral filtering steps

# -- STEP 1 --
# downsample image using Gaussian pyramid 
# 降采样的时候需要注意图片的尺寸，因为有可能恢复的图片和原始图片的尺寸会有一些变化
img_color = img_rgb
for _ in range(numDownSamples):
    img_color = cv2.pyrDown(img_color)

# repeatedly apply small bilateral filter instead of applying
## one large filter
for _ in range(numBilateralFilters):
    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

# upsample image to original size
for _ in range(numDownSamples):
    img_color = cv2.pyrUp(img_color)

# -- STEPS 2 and 3 --
# convert to grayscale and apply median blur
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)

# -- STEP 4 --
# detect and enhance edges
img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 9, 2)

# -- STEP 5 --
# convert back to color so that it can be bit-ANDed with color image
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
cartoon_img = cv2.bitwise_and(img_color, img_edge)
cv2.imshow('cartoon filter',cartoon_img)
cv2.waitKey(0)