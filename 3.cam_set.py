import cv2
import numpy as np

capture = cv2.VideoCapture(1)

# mtx=Camera matrix : Calibration 값 삽입 3*3행렬
# dist=dist : Calibration 값 삽입 1*5행렬

mtx = [[667.94490262 ,  0.     ,    299.06744999],
 [  0.     ,    667.47608479, 230.37419165],
 [  0.      ,     0.    ,       1.        ]]

dist = [[-0.44734875,  0.32597917,  0.0028946,   0.00098143, -0.30192641]]

mtx = np.array(mtx)

dist = np.array(dist)

def mouse_callback(event, x, y, flags, param): 
    print([x,y])
    

cv2.namedWindow('VideoFrame')  #마우스 이벤트 영역 윈도우 생성

cv2.setMouseCallback('VideoFrame', mouse_callback)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()

    img = frame
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dst = cv2.line(dst,(310,240),(330,240),[255,255,255],1)
    dst = cv2.line(dst,(320,230),(320,250),[255,255,255],1)

    

    cv2.imshow("VideoFrame", cv2.resize(dst[44:432 , 27:592], (640,480)))
    #cv2.imshow("VideoFrame",dst)

capture.release()
cv2.destroyAllWindows()

