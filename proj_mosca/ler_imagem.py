import cv2
import imutils

def ler_imagem(caminho):
    # Leitura da imagem
    img = cv2.imread(caminho)

    # Redimensiona a imagem // Tamanho original 4160 x 3120
    #resized = cv2.resize(img, None, fx = 0.40, fy = 0.40, interpolation = cv2.INTER_NEAREST)
    resized = imutils.resize(img, width=2160, inter=cv2.INTER_NEAREST)

    return resized