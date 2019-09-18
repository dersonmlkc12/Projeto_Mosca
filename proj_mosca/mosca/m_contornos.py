import cv2
import numpy as np
from skimage import measure

def regiao_int(imagem,img):
    # Binarizacao da imagem
    ret, thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Rotula as regiões conectadas da matriz(imagem)
    labels = measure.label(thresh, neighbors=8, background=0)

    # Criação de um arranjo de qualquer formato contendo apenas zeros
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Laco sobre os componentes encontrados da imagem
    for label in np.unique(labels):

        # Se este for o rótulo de fundo, ignore-o
        if label == 0: continue

        # Caso contrário, construa a máscara de etiqueta
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        #print("pixels", numPixels)
        # verifica o numero de  pixels de cada componente
        if numPixels > 20 and numPixels < 500:
            mask = cv2.add(mask, labelMask)

    # Mascara de recorte do bovino na imagem de entrada
    ret, thresh1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = cv2.bitwise_and(img, img, mask=thresh1)
    masc = res.copy()

    # Eliminar ruidos de luminosidade
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            (b, g, r) = res[i, j]
            if (r > 127 and g > 127 and  r > 127):
                masc[i, j] = (0, 0, 0)

    masc = cv2.cvtColor(masc, cv2.COLOR_BGR2GRAY)

    return masc