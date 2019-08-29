import cv2
import numpy as np

def watershed(imagem, img, erod, dilat):
    # Aplicação de operações de morfologia matemática na imagem
    erode = cv2.erode(imagem, (np.ones((erod[0], erod[1]), np.uint8)), iterations=erod[2])
    dilate = cv2.dilate(imagem, (np.ones((dilat[0], dilat[1]), np.uint8)), iterations=dilat[2])

    ret, binariza = cv2.threshold(dilate, 1, 127, 1)
    marker = cv2.add(erode, binariza)
    marker32 = np.int32(marker)

    # Aplicacao do algoritmo da Bacia Hidrografica
    markers = cv2.watershed(img, marker32)
    m = cv2.convertScaleAbs(marker32)
    return m,markers