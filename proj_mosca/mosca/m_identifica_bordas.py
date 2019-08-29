import cv2

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