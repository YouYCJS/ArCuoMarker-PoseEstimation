import cv2

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE바
        img_name = "C:/Users/yyc73/Pictures/cam3/cam3_{}.png".format(img_counter)
        # 원하는 파일 경로 세팅 필수!!!!
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
cam.release()

cv2.destroyAllWindows()