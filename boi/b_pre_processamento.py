import cv2

def b_pre_processamento(imagem,s):
    # Transforma a imagem em tons de cinza
    imgCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Histograma da imagem
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgCinza = clahe.apply(imgCinza)

    # Suavizacao da imagem
    imgSuav = cv2.GaussianBlur(imgCinza, (s[0], s[1]), 0)

    return imgSuav
