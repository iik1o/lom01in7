import cv2
import cv2 as cv
import numpy as np
cv2.namedWindow("MyWindow", cv2.WINDOW_NORMAL)
img = cv2.imread("tsq.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,50,255]))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if 1500<cv2.contourArea(cnt)<5000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
cv2.imshow("MyWindow", img)
cv2.waitKey(0)
cv2.destroyAllWindows()