import cv2

def b_pre_processamento(img,s):
    # Transforma a imagem em tons de cinza
    imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histograma da imagem
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    imgCinza = clahe.apply(imgCinza)

    # Suavizacao da imagem
    #imgSuav = cv2.GaussianBlur(imgCinza, (1, 1), 1)
    imgSuav = cv2.bilateralFilter(imgCinza, s[0], s[1], s[2])#Bilateral
    #imgSuav = cv2.medianBlur(imgCinza,3)
    #imgSuav = cv2.blur(imgCinza,(7,7))

    return imgSuav