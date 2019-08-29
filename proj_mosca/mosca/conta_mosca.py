import cv2
import numpy as np
from random import randint

def identifica_bordas(img_bovino):
    # Converte a imagem para tons de cinza
    cinza = cv2.cvtColor(img_bovino, cv2.COLOR_BGR2GRAY)

    # Detector de bordas Sobel
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(cinza, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(cinza, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.3, abs_grad_y, 0.3, 0)
    #cv2.imshow("Sobel", grad)

    return grad

def melhora_imagem(moscas):
    # Binarização
    img2 = cv2.threshold(moscas, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Segmentacao Mosca", img2)

    # Remoção de ruídos
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Encontrar certa área de primeiro plano
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    #cv2.imshow("distance", dist_transform)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    #cv2.imshow("sure_fg", sure_fg)

    sure_fg = np.uint8(sure_fg)

    return sure_fg

def escreve(imgcopy,total):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cor = (255, 10, 0)
    cv2.putText(imgcopy, str(total) + " Moscas do Chifre Encontradas!", (10, 70), fonte, 0.5, cor, 0, cv2.LINE_AA)
    cv2.imshow("Software para contagem de infestacao de moscas do chifre - SOFCIMC", imgcopy)


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

        # remover partes que nao sao moscas (muito grandes)
        if perimeter > 0.1 and perimeter <= 40 and perc < 4:
            # desenha cada retangulo de uma cor aleatoria na imagem final
            # rcolor = (randint(0, 255), randint(0, 255), randint(0, 255))
            rcolor = (255, 0, 0)
            cv2.drawContours(imgcopy, [c], -1, rcolor, 2)
            cv2.rectangle(imgcopy, (x, y), (x + w, y + h), rcolor)
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            cor = (0, 0, 255)
            cv2.putText(imgcopy, str(total), (x, y), fonte, 0.3, cor, 0, cv2.LINE_AA)
            #print("Contorno", total, "perimetro: ", perimeter, "perc: ", perc)
            total += 1

    cv2.imshow("Imagem final", imgcopy)
    #print("contador", total)
    escreve(imgcopy,total)

def conta_mosca(segm,original):
    imgcopy = cv2.imread(original)
    bovino = cv2.resize(segm, (1040, 780), interpolation=cv2.INTER_AREA)
    imgcopy = cv2.resize(imgcopy, (1040, 780), interpolation=cv2.INTER_AREA)

    bov_proc = identifica_bordas(bovino)
    mosca = melhora_imagem(bov_proc)
    identifica_mosca(mosca,imgcopy)
