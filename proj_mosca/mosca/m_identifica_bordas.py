import cv2

def identifica_bordas(img_bovino,x,y):
    # Converte a imagem para tons de cinza
    cinza = cv2.cvtColor(img_bovino, cv2.COLOR_BGR2GRAY)

    # Equaliza Histograma
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    cinza = clahe.apply(cinza)

    cinza = cv2.GaussianBlur(cinza, (3, 3), 0)

    # Detector de bordas Sobel
    ddepth = cv2.CV_64F #CV_16S
    grad_x = cv2.Sobel(cinza, ddepth, x[0], x[1], ksize=x[2], scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(cinza, ddepth, y[0], y[1], ksize=y[2], scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, x[3], abs_grad_y, y[3], 0)

    return grad