import cv2

def identifica_bordas(img_bovino,bord):
    # Converte a imagem para tons de cinza
    cinza = cv2.cvtColor(img_bovino, cv2.COLOR_BGR2GRAY)

    # Suavização
    cinza = cv2.blur(cinza, (3, 3))
    #cinza = cv2.GaussianBlur(cinza, (5, 5), 1)

    # Detecta as bordas
    edges = cv2.Canny(cinza, bord[0], bord[1])

    return edges