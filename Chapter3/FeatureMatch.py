# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:35:16 2018

@author: Bllue
"""



import cv2

img = cv2.imread('deeplearning.jpg')
img = cv2.resize(img, (540, 720))
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#cv2.imshow('pic',img)
#cv2.waitKey(0)

min_hessian = 2000
SURF = cv2.xfeatures2d.SURF_create(min_hessian)
key_query, desc_query = SURF.detectAndCompute(img,None)

imgout = cv2.drawKeypoints(img,key_query,img)

#cv2.imshow('detect', imgout)
#cv2.waitKey(0)

frame = cv2.imread('frame.jpg')
frame = cv2.resize(frame, (540, 720))
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
key_train, desc_train = SURF.detectAndCompute(frame,None)

#cv2.imshow('frame', frame)
#cv2.waitKey(0)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


matches = flann.knnMatch(desc_query, desc_train, k=2)


good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        
img_match = cv2.drawMatchesKnn(img, key_query, frame, key_train, [good_matches], None,flags=2)
cv2.imshow('match',img_match)
cv2.waitKey(0)


# 书里面 关于Homography estimation 的代码复现之后结果不对，需要后续研究一下

# reference https://www.jianshu.com/p/1f6195352b26