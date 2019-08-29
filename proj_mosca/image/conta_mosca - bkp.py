import cv2
import numpy as np
from random import randint

def conta_mosca(segm,original):
    imgcopy = cv2.imread(original)

    mosca = cv2.resize(segm, (1280, 768), interpolation=cv2.INTER_AREA)
    imgcopy = cv2.resize(imgcopy, (1280, 768), interpolation=cv2.INTER_AREA)

    # Converte a imagem para tons de cinza
    cinza = cv2.cvtColor(mosca, cv2.COLOR_BGR2GRAY)

    # Equaliza Histograma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cinza = clahe.apply(cinza)

    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)

    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(cinza, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(cinza, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.3, abs_grad_y, 0.3, 0)
    cv2.imshow("grad", grad)

    img2 = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Segmentacao Mosca", img2)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel, iterations=1)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    cv2.imshow("sure_bg", sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    cv2.imshow("distance", dist_transform)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    cv2.imshow("sure_fg", sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow("unknow", unknown)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # cv2.imshow("markers",markers)

    markers = cv2.watershed(mosca, markers)

    m = cv2.convertScaleAbs(markers)
    m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mosca[markers == -1] = [255, 0, 0]

    cv2.imshow("mosca80", mosca)

    r, contours, hi = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    total = 0;
    # numero de objetos detectados dentro da imagem
    count2 = 0
    # verificar contornos
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        c = cv2.approxPolyDP(cnt, epsilon, True)
        (x, y, w, h) = cv2.boundingRect(c)
        # porcentagem do retangulo preenchido pelo contorno
        perc = w / h if h > 0 else 0;
        perimeter = cv2.arcLength(cnt, True)

        # remover partes que nao sao moscas (muito grandes)
        # if perimeter > 3 and perimeter < 16 and perc > 0.7 and perc < 5:
        if perimeter > 0.1 and perimeter < 21:
            # desenha cada retangulo de uma cor aleatoria na imagem final
            rcolor = (randint(0, 255), randint(0, 255), randint(0, 255))
            cv2.drawContours(imgcopy, [c], -1, rcolor, 2)
            cv2.rectangle(imgcopy, (x, y), (x + w, y + h), rcolor)
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            cor = (255, 255, 255)
            # if total == 45:
            cv2.putText(imgcopy, str(total), (x, y), fonte, 0.3, cor, 0, cv2.LINE_AA)
            # print("Posição", total, "perimetro: ", perimeter, "perc: ", perc)
            count2 += 1
            total += 1

    cv2.imshow("Imagem final", imgcopy)
    print("contador", count2)