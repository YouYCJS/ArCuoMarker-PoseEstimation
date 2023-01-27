import cv2

img=cv2.imread('C:/Users/yyc73/Desktop/ArUCo-Markers-Pose-Estimation-Generation-Python-main/Images/pose_output_image.png')

x_pos,y_pos,width,height=cv2.selectROI("location",img,False)
print("x,y :",x_pos,y_pos)
print("w,h :",width,height)

cv2.destroyAllWindows()