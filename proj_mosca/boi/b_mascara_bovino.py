import cv2

def mascara_bovino(water,water1,img):
    ret, thresh = cv2.threshold(water, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Mascara de recorte do bovino na imagem de entrada
    res = cv2.bitwise_and(img, img, mask=thresh)
    res[water1 == -1] = [255, 255, 255]

    return res