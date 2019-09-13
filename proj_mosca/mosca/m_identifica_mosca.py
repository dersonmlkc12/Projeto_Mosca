import cv2
import numpy as np

def identifica_mosca(labels,img_bovino,img):
    total = 0
    for label in np.unique(labels):
        if label == 0:continue

        mask = np.zeros(img_bovino.shape,dtype="uint8")
        mask[labels == label] = 255
        r, cnts, hi = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cn in cnts:
            epsilon = 0.025 * cv2.arcLength(cn, True)
            c = cv2.approxPolyDP(cn, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(c)

            perc = w / h if h > 0 else 0;
            perimeter = cv2.arcLength(cn, True)

            if perimeter > 0 and perimeter <= 40:
                c = max(cnts, key=cv2.contourArea)
                ((x,y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(img, (int(x),int(y)),int(r),(0, 0, 255),2)
                #cv2.putText(img, str(total), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), cv2.LINE_AA)
                total += 1

    return img, total