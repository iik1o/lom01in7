import cv2
import numpy as np

K = np.array([[718.8, 0, 505.75], [0, 764.67, 228.17], [0, 0, 1]], dtype=np.float32)
dist = np.array([-0.672, 0.775, 0.0159, -0.0811, -0.327])

obj = np.float32([[-25, -25, 0], [25, -25, 0], [25, 25, 0], [-25, 25, 0]])
axis = np.float32([[30, 0, 0], [0, 30, 0], [0, 0, 30]])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)   
 
        image_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(obj, image_pts, K, dist)
        axis_img, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
        center, _ = cv2.projectPoints(np.array([[0,0,0]]), rvec, tvec, K, dist)
        ctr = tuple(center.ravel().astype(int))

        cv2.line(frame, ctr, tuple(axis_img[0].ravel().astype(int)), (0,0,255), 3)
        cv2.line(frame, ctr, tuple(axis_img[1].ravel().astype(int)), (0,255,0), 3)
        cv2.line(frame, ctr, tuple(axis_img[2].ravel().astype(int)), (255,0,0), 3)

    cv2.imshow('pose', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
