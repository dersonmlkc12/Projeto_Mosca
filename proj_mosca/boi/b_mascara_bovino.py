import cv2

def mascara_bovino(water,water1,img):
    ret, thresh = cv2.threshold(water, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Mascara de recorte do bovino na imagem de entrada
    res = cv2.bitwise_and(img, img, mask=thresh)
    res[water1 == -1] = [255, 255, 255]

    masc = res.copy()

    # Eliminar ruidos de luminosidade
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            (b, g, r) = res[i, j]
            if (r > 240 and g > 240 and b > 240):
                masc[i, j] = (0, 0, 0)
    return masc