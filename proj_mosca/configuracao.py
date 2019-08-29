import cv2
import numpy as np

def configuracao(conf):
    largura = 768
    altura = 768

    img = np.zeros((largura,altura,3),dtype=np.uint8)

    cv2.rectangle(img,(0,0),(largura,altura),(255,255,255), -1)
    #cv2.rectangle(img,(0,0),(largura-5,altura-5),(255,255,255),5)

    for i in range (4):
        fonte = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,conf[i],(10,(1+i)*70), fonte, 0.95, (0,0,0), 1, cv2.LINE_AA)

    return img