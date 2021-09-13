import cv2
import numpy as np
from skimage import measure

def regiao_interesse(imagem):
    # Binarizacao da imagem
    ret, thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow("Imagem binarizada com segmentacao do bovino",thresh)

    # Rotula as regiões conectadas da matriz(imagem)
    labels = measure.label(thresh, neighbors=8, background=0)

    # Criação de um arranjo de qualquer formato contendo apenas zeros
    mask = np.zeros(thresh.shape, dtype="uint8")
    cont = 20000

    # Laco sobre os componentes encontrados da imagem
    for label in np.unique(labels):

        # Se este for o rótulo de fundo, ignore-o
        if label == 0: continue

        # Caso contrário, construa a máscara de etiqueta
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # verifica o componente com o maior numero de  pixels
        if numPixels > 120000:
            if numPixels >= cont:
                mask = cv2.add(mask, labelMask)
                cont = numPixels

    return mask