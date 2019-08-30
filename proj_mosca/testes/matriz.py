import cv2
import csv

img = cv2.imread('teste.jpg')

# Redimensiona a imagem
img = cv2.resize(img, (1040, 780), interpolation=cv2.INTER_NEAREST)#INTER_AREA

altura = img.shape[0]
largura = img.shape[1]

mat = []

with open("matriz.csv",'w',newline='') as saida:
    escrever = csv.writer(saida,delimiter=';')

    for i in range(0, altura):
        linha = []
        for j in range(0, largura):
            (b, g, r) = img[i, j]
            linha.append(b)
        escrever.writerow([linha])

#print(mat)

cv2.waitKey(0)
cv2.destroyAllWindows()