import numpy as np
import cv2
from skimage import measure

def escreve(imgcopy,total):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cor = (255, 10, 0)
    cv2.putText(imgcopy, str(total) + " Moscas do Chifre Encontradas!", (10, 70), fonte, 0.5, cor, 0, cv2.LINE_AA)
    cv2.imshow("Software para contagem de infestacao de moscas do chifre - SOFCIMC", imgcopy)

def pre_processa(segm):
    gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)

    # Convolução 2D (Filtragem de Imagem)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgConv = cv2.filter2D(gray, -1, kernel)
    sharp = np.float32(gray)
    imgResult = sharp - imgConv
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    #cv2.imshow("Imagem com convolucao",imgResult)

    # Suavizacao da imagem
    imgSuav = cv2.GaussianBlur(imgResult, (3, 3), 0)
    # cv2.imshow("Imagem com suavizacao",imgSuav)
    #cv2.imshow("proc", imgSuav)

    # Detector de bordas Sobel
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(imgSuav, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(imgSuav, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    ret, thresh = cv2.threshold(grad,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow("thresh",thresh)

    return thresh

def limpa_imagem(imgp):
    # Rotula as regiões conectadas da matriz(imagem)
    labels = measure.label(imgp, neighbors=8, background=0)

    # Criação de um arranjo de qualquer formato contendo apenas zeros
    mask = np.zeros(imgp.shape, dtype="uint8")

    # Laco sobre os componentes encontrados da imagem
    for label in np.unique(labels):

        # Se este for o rótulo de fundo, ignore-o
        if label == 0: continue

        # Caso contrário, construa a máscara de etiqueta
        labelMask = np.zeros(imgp.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # verifica o componente com o maior numero de  pixels
        if numPixels >= 15 and numPixels <= 120:
            #print("pixels: ", numPixels)
            mask = cv2.add(mask, labelMask)
        #cv2.imshow("mask",mask)
        #cv2.imshow("original",imgcopy)
    return mask

def id_mosca(mosca,imgcopy):
    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opening = cv2.erode(mosca,kernel,iterations = 1)
    #opening = cv2.morphologyEx(mosca, cv2.MORPH_CLOSE, kernel, iterations=1)
    #cv2.imshow("opening",opening)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    #sure_bg = cv2.morphologyEx(mosca, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    #cv2.imshow("sure",sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(imgcopy, markers)
    imgcopy[markers == -1] = [255, 0, 0]

    cv2.imshow("Imagem contornos",imgcopy)


def identifica_mosca(mosca,imgcopy):
    r, contours, hi = cv2.findContours(mosca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # numero de objetos detectados dentro da imagem
    total = 0;

    # verificar contornos
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        c = cv2.approxPolyDP(cnt, epsilon, True)
        (x, y, w, h) = cv2.boundingRect(c)
        # porcentagem do retangulo preenchido pelo contorno
        perc = w / h if h > 0 else 0;
        perimeter = cv2.arcLength(cnt, True)
        print("Contorno", total, "perimetro: ", perimeter, "perc: ", perc)
        # remover partes que nao sao moscas (muito grandes)
        if perimeter > 0.1 and perimeter <= 60 and perc < 15:
            # desenha cada retangulo de uma cor aleatoria na imagem final
            # rcolor = (randint(0, 255), randint(0, 255), randint(0, 255))
            rcolor = (255, 0, 0)
            cv2.drawContours(imgcopy, [c], -1, rcolor, 2)
            cv2.rectangle(imgcopy, (x, y), (x + w, y + h), rcolor)
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            cor = (0, 0, 255)
            cv2.putText(imgcopy, str(total), (x, y), fonte, 0.3, cor, 0, cv2.LINE_AA)

            total += 1

    # cv2.imshow("Imagem final", imgcopy)
    #print("contador", total)
    escreve(imgcopy,total)


def conta_mosca(segm,original):
    imgcopy = cv2.imread(original)
    imgcopy = cv2.resize(imgcopy, (832, 624), interpolation=cv2.INTER_AREA)
    im = pre_processa(segm)
    im1 = limpa_imagem(im)
    #identifica_mosca(im1,imgcopy)
    id_mosca(im1, imgcopy)