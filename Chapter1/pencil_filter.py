# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:13:36 2018

@author: Bllue
"""

import cv2


img_rgb = cv2.imread('me.jpg')
img_rgb = cv2.resize(img_rgb, (540, 720))
cv2.imshow('origin',img_rgb)

bg_gray='pencilsketch_bg.jpg'
canvas = cv2.imread(bg_gray, cv2.CV_8UC1)

canvas = cv2.resize(canvas, (540, 720))
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
img_blend = cv2.divide(img_gray, img_blur, scale=256)
img_blend = cv2.multiply(img_blend, canvas, scale=1./256)
pencil_img = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)
cv2.imshow('pencil filter',pencil_img)
cv2.waitKey(0)