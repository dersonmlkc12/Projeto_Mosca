import cv2
import numpy
from scipy.ndimage import label

def m_watershed(img_bovino,img):

    img_b = cv2.morphologyEx(img_bovino,cv2.MORPH_OPEN,(numpy.ones((3,3),numpy.uint8)),iterations=1)
    cv2.imshow("img_b",img_b)

    border = cv2.dilate(img_b,None,iterations=2)
    cv2.imshow("dilate", border)

    border = border - cv2.erode(border,None)
    cv2.imshow("erode", border)

    #dt = cv2.distanceTransform(img_b, cv2.DIST_L2, 3)
    dt = cv2.distanceTransform(img_b,1,5)
    cv2.imshow("distance", dt)

    lb1, ncc = label(dt)
    lb1 = lb1 * (255 / ncc)
    cv2.imshow("lb1", lb1)

    lb1[border == 255] = 255
    cv2.imshow("lb1_", lb1)

    lb1 = lb1.astype(numpy.int32)
    w = cv2.watershed(img,lb1)

    lb1 [w == -1] = 0
    lb1 = lb1.astype(numpy.uint8)
    teste = 255 - lb1
    cv2.imshow("teste",teste)

    img[teste == 255] = (0,0,255)
    cv2.imshow("img",img)

    return teste