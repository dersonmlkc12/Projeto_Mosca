import cv2
def ler_imagem(caminho):
    # Leitura da imagem
    img = cv2.imread(caminho)

    # Redimensiona a imagem
    img = cv2.resize(img, (832, 624), interpolation=cv2.INTER_NEAREST)#INTER_AREA
    return img

def exp1(image):

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            (b, g, r) = image[i, j]
            if r > 10 and r < 41 and g > 4 and g < 40 and b > 0 and b < 40:
            #if r < 80:
                image[i, j] = (0, 0, 0)
            #else:
                #image[i, j] = (0, 0, 0)
    cv2.imshow("Final Image", image)
    return image

def exper(original):
    original = ler_imagem(original)
    teste = exp1(original)
