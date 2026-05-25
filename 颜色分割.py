import cv2
import cv2 as cv
import numpy as np

cv2.namedWindow("MyWindow", cv2.WINDOW_NORMAL)
img = cv.imread("bed.jpg")
img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
low = np.array([120,20,60])
upper = np.array([180,255,255])
mask = cv2.inRange(img2, low, upper)
kernel = np.ones((3,3),np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
res = cv.bitwise_and(img, img, mask=mask)
cv.imshow("MyWindow",res)
cv.waitKey(0)
cv.destroyAllWindows()