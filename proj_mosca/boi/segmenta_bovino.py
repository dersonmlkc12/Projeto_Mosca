import cv2
import numpy as np
from skimage import measure

def ler_imagem(caminho):
    # Leitura da imagem
    img = cv2.imread(caminho)

    # Redimensiona a imagem
    img = cv2.resize(img, (832, 624), interpolation=cv2.INTER_NEAREST)#INTER_AREA 1040, 780
    return img

def pre_processamento(imagem):

    # Transforma a imagem em tons de cinza
    imgCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgCinza = clahe.apply(imgCinza)

    # Suavizacao da imagem
    imgSuav = cv2.GaussianBlur(imgCinza, (7, 7), 0)
    # cv2.imshow("Imagem com suavizacao",imgSuav)
    #cv2.imshow("proc",imgSuav)
    return imgSuav

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
        #print("pixels: ", numPixels)
        # verifica o componente com o maior numero de  pixels
        if numPixels > 120000:
            if numPixels >= cont:
                mask = cv2.add(mask, labelMask)
                cont = numPixels
                #print("pix: ", cont)
    return mask

def mascara_bovino(imagem,img):
    # Aplicação de operações de morfologia matemática na imagem
    erode = cv2.erode(imagem, (np.ones((3, 3), np.uint8)), iterations=2)
    dilate = cv2.dilate(imagem, (np.ones((7, 7), np.uint8)), iterations=10)
    ret, binariza = cv2.threshold(dilate, 1, 140, 1)
    marker = cv2.add(erode, binariza)
    marker32 = np.int32(marker)

    # Aplicacao do algoritmo da Bacia Hidrografica
    markers = cv2.watershed(img, marker32)
    m = cv2.convertScaleAbs(marker32)
    #cv2.imshow("Resultado da imagem apos aplicacao do algoritmo da Bacia Hidrografica",m)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Mascara de recorte do bovino na imagem de entrada
    res = cv2.bitwise_and(img, img, mask=thresh)
    #cv2.imshow("Mascara da segmentacao aplicada a imagem original", res)
    res[markers == -1] = [255, 255, 255]
    cv2.imshow("imagem com linha de segmentacao", res)
    return res

def segmenta_bovino(caminho):
    imagem = ler_imagem(caminho)
    pre = pre_processamento(imagem)
    regiao = regiao_interesse(pre)
    mascara = mascara_bovino(regiao,imagem)

    return mascara