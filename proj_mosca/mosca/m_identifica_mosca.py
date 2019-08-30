import cv2

def identifica_mosca(mosca,imgcopy):
    r, contours, hi = cv2.findContours(mosca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # numero de objetos detectados dentro da imagem
    total = 0;

    # verificar contornos
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        c = cv2.approxPolyDP(cnt, epsilon, True)
        (x, y, w, h) = cv2.boundingRect(c)

        # porcentagem do retangulo preenchido pelo contorno
        perc = w / h if h > 0 else 0;
        perimeter = cv2.arcLength(cnt, True)

        # remover partes que nao sao moscas (muito grandes)
        if perimeter > 0.1 and perimeter <= 40 and perc < 4:
            # desenha cada retangulo de uma cor aleatoria na imagem final
            # rcolor = (randint(0, 255), randint(0, 255), randint(0, 255))
            rcolor = (255, 0, 0)
            cv2.drawContours(imgcopy, [c], -1, rcolor, 2)
            cv2.rectangle(imgcopy, (x, y), (x + w, y + h), rcolor)
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            cor = (0, 0, 255)
            #cv2.putText(imgcopy, str(total), (x, y), fonte, 0.3, cor, 0, cv2.LINE_AA)
            #print("Contorno", total, "perimetro: ", perimeter, "perc: ", perc)
            total += 1

    #cv2.imshow("Imagem final", imgcopy)
    #print("contador", total)

    return imgcopy,total