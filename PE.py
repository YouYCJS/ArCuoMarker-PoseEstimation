import cv2

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()

    frame = cv2.line(frame,(0,240),(640,240),[255,255,255],1)
    frame = cv2.line(frame,(320,0),(320,480),[255,255,255],1)

    cv2.imshow("VideoFrame", frame)
    
    

capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()