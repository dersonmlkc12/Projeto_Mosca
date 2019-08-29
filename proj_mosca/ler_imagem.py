import cv2

def ler_imagem(caminho):
    # Leitura da imagem
    img = cv2.imread(caminho)

    # Redimensiona a imagem
    img = cv2.resize(img, (1040, 780), interpolation=cv2.INTER_NEAREST)#INTER_AREA 1040, 780
    return img