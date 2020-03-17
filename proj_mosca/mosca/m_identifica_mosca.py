import cv2

def identifica_mosca(img_bovino,img,per):
    teste = img_bovino.copy()
    teste = cv2.cvtColor(teste, cv2.COLOR_GRAY2RGB)

    r, contours, hi = cv2.findContours(img_bovino, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # numero de objetos detectados dentro da imagem
    total = 0;
    horn = []

    # verificar contornos
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        c = cv2.approxPolyDP(cnt, epsilon, True)
        (x, y, w, h) = cv2.boundingRect(c)

        # porcentagem do retangulo preenchido pelo contorno
        perc = w / h if h > 0 else 0;
        perimeter = cv2.arcLength(cnt, True)
        #print("per: ",perc," perime: ",perimeter)

        # remover partes que nao sao moscas (muito grandes)
        if perimeter >= per[0] and perimeter < per[1] and perc > 0 and perc <= 1.9:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), 2)
            cv2.circle(teste, (int(x), int(y)), int(r), (0, 0, 255), 1)
            #print((str(total)) + ' indent: '+str(int(x))+','+str(int(y)))
            total += 1
            #print(total, " - peri: ", perimeter)
            #x = int(x)
            #y = int(y)
            horn.append([int(x),int(y)])

    return img, total, teste, horn