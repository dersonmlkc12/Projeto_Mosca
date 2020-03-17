import cv2
import numpy as np
from skimage import measure

def regiao_interesse(imagem):
    # Binarizacao da imagem
    ret, thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Rotula as regiões conectadas da matriz(imagem)
    labels = measure.label(thresh, neighbors=4, background=0)

    # Criação de um arranjo de qualquer formato contendo apenas zeros
    mask = np.zeros(thresh.shape, dtype="uint8")

    cont = 0
    temp = ''

    # Laco sobre os componentes encontrados da imagem
    for label in np.unique(labels):
        # Se este for o rótulo de fundo, ignore-o
        if label == 0: continue

        # Caso contrário, construa a máscara de etiqueta
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # verifica o componente com o maior numero de  pixels
        if numPixels >= cont:
                temp = labelMask
                cont = numPixels

    mask = cv2.add(mask, temp)

    return mask