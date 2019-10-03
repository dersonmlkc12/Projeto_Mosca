import cv2
import numpy as np
import imutils
from skimage import measure

def identifica_mosca(img_bovino,img,per):
    teste = img_bovino.copy()
    teste = cv2.cvtColor(teste, cv2.COLOR_GRAY2RGB)

    r, contours, hi = cv2.findContours(img_bovino, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # numero de objetos detectados dentro da imagem
    total = 0;

    # verificar contornos
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        c = cv2.approxPolyDP(cnt, epsilon, True)
        (x, y, w, h) = cv2.boundingRect(c)

        # porcentagem do retangulo preenchido pelo contorno
        #perc = w / h if h > 0 else 0;
        perimeter = cv2.arcLength(cnt, True)
        #print("per: ",perc," perime: ",perimeter)

        # remover partes que nao sao moscas (muito grandes)
        if perimeter >= per[0] and perimeter < per[1]:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), 2)
            cv2.circle(teste, (int(x), int(y)), int(r), (0, 0, 255), 1)
            #if total == 33:
            #cv2.putText(img, str(total), (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            #cv2.putText(teste, str(total), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            total += 1
            #print(total, " - peri: ", perimeter)
            #print(total)
    #teste1 = imutils.resize(img, width=2160, inter=cv2.INTER_NEAREST)
    #cv2.imshow("moscas",teste1)

    return img, total, teste