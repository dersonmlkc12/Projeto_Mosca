import cv2
import numpy as np
import imutils

def identifica_mosca(lab,img_bovino,img):
    total = 0
    for label in np.unique(lab):
        if label == 0: continue

        mask = np.zeros(img_bovino.shape, dtype="uint8")
        mask[lab == label] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        perimeter = cv2.arcLength(c, True)

        if perimeter > 10 and perimeter < 35:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), 2)
            #cv2.putText(img, str(total), (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            #print(total, " - peri: ", perimeter)
            total += 1

    return img, total