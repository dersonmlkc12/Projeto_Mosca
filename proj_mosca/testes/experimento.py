import numpy as np
import cv2
from datetime import datetime
from skimage import measure
import imutils

data = datetime.now()
fname = data.strftime('%Y-%m-%d-%H-%M-%S')

img1 = '../image_teste/img32.jpg'
orig = cv2.imread(img1)



orig = imutils.resize(orig, width=2773, inter=cv2.INTER_NEAREST)

org = adjust_gamma(orig)

#Tons Cinza
imgCinza = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
imgCinza = clahe.apply(imgCinza)

#Suavização
imgCinza = cv2.bilateralFilter(imgCinza, 5, 25, 25)#Bilateral

########################################################################################################
ret, thresh = cv2.threshold(imgCinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
##################################################################################

# Aplicação de operações de morfologia matemática na imagem
erode = cv2.erode(mask, (np.ones((5,5), np.uint8)), iterations=1)
dilate = cv2.dilate(mask, (np.ones((3,3), np.uint8)), iterations=4)

ret, binariza = cv2.threshold(dilate, 1, 127, 1)
marker = cv2.add(erode, binariza)
marker32 = np.int32(marker)

# Aplicacao do algoritmo da Bacia Hidrografica
markers = cv2.watershed(orig, marker32)
m = cv2.convertScaleAbs(marker32)

cv2.imwrite('../resultados/res_tes'+fname+'.jpg', m)

