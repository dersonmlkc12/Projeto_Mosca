import cv2

def escreve(imgcopy,total):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cor = (0, 0, 255)
    res = cv2.putText(imgcopy, str(total) + " Moscas do Chifre Encontradas!", (10, 70), fonte, 0.95, cor, 0, cv2.LINE_AA)
    #cv2.imshow("Software para contagem de infestacao de moscas do chifre - SOFCIMC", imgcopy)
    return res