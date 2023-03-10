'''
Sample Usage:-
python 4.pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_6X6_1000
'''
import math
from imutils.video import FPS
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video streammatrix_coefficients
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)
	
        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,distortion_coefficients)
            
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 
            print(corners[i])
            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
    return frame

if __name__ == '__main__':
    aruco_dict_type = ARUCO_DICT["DICT_6X6_100"]
    k = np.load('C:/Users/yyc73/Desktop/ArUCo-Markers-Pose-Estimation-Generation-Python-main/calibration_matrix.npy')
    d = np.load('C:/Users/yyc73/Desktop/ArUCo-Markers-Pose-Estimation-Generation-Python-main/distortion_coefficients.npy')

    video = cv2.VideoCapture(1)

    time.sleep(2.0)

    
    while True:
        ret, frame = video.read()

        if not ret:
            break

        #????????? ?????? ??????
        h,  w = frame.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(k,d,(w,h),1,(w,h))
        dst = cv2.undistort(frame, k, d, None, newcameramtx)
        calibrated_frame = cv2.resize(dst[41:443 , 32:601], (640,480))
        
        #????????? ?????? ?????? ??? ??? ?????????
        pose_esitmation(calibrated_frame, aruco_dict_type, k, d)

        calibrated_frame = cv2.line(calibrated_frame,(0,240),(640,240),[255,255,255],1)
        calibrated_frame = cv2.line(calibrated_frame,(320,0),(320,480),[255,255,255],1)
        #fps.update()

        cv2.imshow("VideoFrame", calibrated_frame)

        
        #?????? ??????(?????? ?????? ??????)
        (x_n,y_n,z_n)=(0.426,0,0.444) #?????? ?????????????????????...m??????

        #ArUco Marker ??? ??????
        #(x_0,y_0,z_0)=(, , 0) #0???
        #(x_1,y_1,z_1)=(, , 0) #1???
        #(x_2,y_2,z_2)=(, , 0) #2???
        #(x_3,y_3,z_3)=(, , 0) #3???
        #(x_4,y_4,z_4)=(, , 0) #4???
        #(x_5,y_5,z_5)=(, , 0) #5???
        #(x_6,y_6,z_6)=(, , 0) #6???
        #(x_7,y_7,z_7)=(, , 0) #7???
        #(x_8,y_8,z_8)=(, , 0) #8???
        #(x_9,y_9,z_9)=(, , 0) #9???
        #(x_10,y_10,z_10)=(, , 0) #10???
        #(x_11,y_11,z_11)=(, , 0) #11???
        #(x_12,y_12,z_12)=(, , 0) #12???
        #(x_13,y_13,z_13)=(, , 0) #13???
        (x_14,y_14,z_14)=(1,1,0) #14???

        #??????=640*480(??????*??????) = 1?????? ???, ??? m??
        w_pix= 0.00065625 #?????? ??????/m(??????) ??????=42cm=0.42m
        h_pix= 0.00059375 #?????? ??????/m(??????) ??????=28.5cm=0.285m

        
        # if ids==14:
        #     print("????????? ?????? : ",ids,"??? / ","????????? ???????????? ?????? : ", x_n-x_14,y_n-y_14,0.444)
        # else:
        #     print("?????? ?????? : ",x_n,y_n,z_n)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    #fps.stop()
    video.release()
    cv2.destroyAllWindows()