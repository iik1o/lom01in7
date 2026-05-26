import cv2
import numpy as np
K = np.array([[718.80106938, 0, 505.74855641],
             [0, 764.67086684, 228.16567087],
             [0, 0, 1]], np.float32)
dist = np.array([[-0.67217889, 0.77494974, 0.01589332, -0.08113757, -0.32714995]], np.float32)

obj = np.array([[0,0,0],[50,0,0],[0,50,0],[50,50,0]], np.float32)
axis = np.array([[40,0,0],[0,40,0],[0,0,40]], np.float32)

cap = cv2.VideoCapture(0)
while True:
    ret, frm = cap.read()
    if not ret: break
    gray = cv2.GaussianBlur(cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY),(3,3),0)
    harris = cv2.dilate(cv2.cornerHarris(gray,2,3,0.04),None)
    y,x = np.where(harris > 0.01*harris.max())
    pts = np.column_stack((x,y)).astype(np.float32)

    if len(pts)==4:
        top4 = pts[np.argsort(harris[y,x])[-4:]]
        tl,br = top4[np.argmin(top4.sum(1))], top4[np.argmax(top4.sum(1))]
        tr,bl = top4[np.argmin(np.diff(top4,1))], top4[np.argmax(np.diff(top4,1))]
        img_pts = np.array([tl,tr,bl,br],np.float32)

        for p in img_pts:
            cv2.circle(frm,(int(p[0]),int(p[1])),5,(255,0,0),-1)

        _,rvec,tvec = cv2.solvePnP(obj,img_pts,K,dist)
        ax_pts,_ = cv2.projectPoints(axis,rvec,tvec,K,dist)
        ax_pts = ax_pts.astype(int)
        o = tuple(img_pts[0].astype(int))
        cv2.line(frm,o,tuple(ax_pts[0][0]),(0,0,255),2)
        cv2.line(frm,o,tuple(ax_pts[1][0]),(0,255,0),2)
        cv2.line(frm,o,tuple(ax_pts[2][0]),(255,255,0),2)

    cv2.imshow("Pose",frm)
    if cv2.waitKey(1): break
cap.release()
cv2.destroyAllWindows()
