import cv2
import numpy as np

def melhora_imagem(moscas):

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(moscas, cv2.MORPH_CLOSE, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(opening, 0.1 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    return sure_fg

