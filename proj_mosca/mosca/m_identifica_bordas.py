import cv2
import numpy as np

def identifica_bordas(img_bovino):
    # Converte a imagem para tons de cinza
    cinza = cv2.cvtColor(img_bovino, cv2.COLOR_BGR2GRAY)

    # Suavização
    blur = cv2.GaussianBlur(cinza,(41,41),0)

    # Passa Alta
    filtered = cinza - blur
    filtered = filtered + 127 * np.ones(cinza.shape, np.uint8)

    res = filtered.copy()

    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            (r) = res[i, j]
            if (r < 105):
                res[i, j] = (0)
            else:
                res[i, j] = (255)

    return filtered, res