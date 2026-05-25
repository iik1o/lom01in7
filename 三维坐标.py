import cv2
import numpy as np
cameraMatrix = np.array([
    [718.8, 0, 505.75],
    [0, 764.67, 228.17],
    [0, 0, 1]
], dtype=np.float32)
distCoeffs = np.array([-0.672, 0.775, 0.0159, -0.0811, -0.327], dtype=np.float32)
objectPoints = np.array([
    [-25, -25, 0],
    [ 25, -25, 0],
    [ 25,  25, 0],
    [-25,  25, 0]
], dtype=np.float32)
axisPoints = np.array([[30,0,0],[0,30,0],[0,0,30]], dtype=np.float32)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 500:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int32)
            pts = box.astype(np.float32)
            pts = pts[np.lexsort((pts[:,0], pts[:,1]))]
            left_up_pos   = pts[0]
            right_up_pos  = pts[1]
            right_down_pos= pts[3]
            left_down_pos = pts[2]
            imagePoints = np.array([
                [left_up_pos[0],    left_up_pos[1]],
                [right_up_pos[0],   right_up_pos[1]],
                [right_down_pos[0], right_down_pos[1]],
                [left_down_pos[0],  left_down_pos[1]]
            ], dtype=np.float32).reshape(4,1,2)
            _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
            if rvec is not None:
                proj, _ = cv2.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs)
                proj = proj.reshape(3,2).astype(int)
                origin = tuple(left_up_pos.astype(int))
                cv2.line(frame, origin, tuple(proj[0]), (0,0,255), 3)
                cv2.line(frame, origin, tuple(proj[1]), (0,255,0), 3)
                cv2.line(frame, origin, tuple(proj[2]), (255,0,0), 3)
    cv2.imshow("solvePnP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
