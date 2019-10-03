import cv2
import numpy as np

def b_pre_processamento(img,s):
    # Transforma a imagem em tons de cinza
    imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histograma da imagem
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgCinza = clahe.apply(imgCinza)

    # Suavizacao da imagem
    #imgSuav = cv2.GaussianBlur(imgCinza, (3, 3), 1)
    imgSuav = cv2.bilateralFilter(imgCinza, 7, 25, 25)#Bilateral

    return imgSuav