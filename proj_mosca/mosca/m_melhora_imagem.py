import cv2
import numpy as np

def melhora_imagem(moscas):
    # Binarização
    img2 = cv2.threshold(moscas, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Remoção de ruídos
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Encontrar certa área de primeiro plano
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    return sure_fg