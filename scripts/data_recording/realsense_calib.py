import cv2 as cv
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np

import argparse
# import rosbag
# import rospy

def realsense_calibrate(frames_folder):

    c1_images = []


    bridge = CvBridge()
    data = np.load(frames_folder, allow_pickle = True)
    count = 0
    for img in data:
        c1_images.append(img)


    #change this if stereo calibration not good.
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 6 #number of checkerboard rows.
    columns = 7 #number of checkerboard columns.
    world_scaling = 0.05 #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    pattern = (6, 7)

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1 in c1_images:
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
       
        c_ret1, corners1 = cv.findChessboardCorners(gray1, pattern, None)

 
        if c_ret1 == True :
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

 
            #print(corners1)
            cv.drawChessboardCorners(gray1, pattern, corners1, c_ret1)
            cv.imshow('img', gray1)
            k = cv.waitKey(100)
            objpoints.append(objp)
            imgpoints_left.append(corners1)

    ret1, mtx1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(objpoints, imgpoints_left, (width, height), None, None)
 
    return ret1, mtx1, dist1



#camera matrix, distortion coefficients
ret, mtx, dist = realsense_calibrate( './head/1.npy')
print("ret: ", ret)
print("mtx: ", mtx)
print("dist: ", dist)

# ret, mtx, dist = realsense_calibrate( './right/2.npy')
# print("ret: ", ret)
# print("mtx: ", mtx)
# print("dist: ", dist)
