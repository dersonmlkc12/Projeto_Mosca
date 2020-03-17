import cv2
import numpy as np
from skimage import measure

def melhora_imagem(moscas):
    ret, thresh = cv2.threshold(moscas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Rotula as regiões conectadas da matriz(imagem)
    labels = measure.label(thresh, neighbors=4, background=255)
    cont = 0

    # Laco sobre os componentes encontrados da imagem
    for label in np.unique(labels):
        if label == 0: continue
        cont += 1

    #print("cont: ",cont)
    #and cont < 40000
    if cont > 19000 :
        itera = 2

    else:
        itera = 1


    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(moscas, cv2.MORPH_CLOSE, kernel, iterations=itera)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(opening, 0.1 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    return sure_fg, cont

