import xml.etree.ElementTree as ET
import numpy as np
import csv

def centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    centroid_x = sum(x_coords) / _len
    centroid_y = sum(y_coords) / _len

    return centroid_x, centroid_y

def valida(moscas,img):
    # Captura o nome do arquivo correspondente para validação
    i = img.split("/")
    i = i[1].split(".")
    caminho = 'annotations/'+i[0]+'.xml'

    tree = ET.parse(caminho)
    root = tree.getroot()

    centroids = []
    for poly in root.iter('polygon'):
        points = []
        for pt in poly.iter('pt'):
            x = int(pt.find('x').text)*0.60
            y = int(pt.find('y').text)*0.60
            points.append([x, y])

        cx, cy = centroid(points)
        centroids.append([cx, cy])

    #print('teste',centroids)

    verdadeiro_pos = 0
    #verdadeiro_neg = 0
    falso_negativo = 0
    #falso_positivo = 0
    output = []
    array_predicted = moscas #valor dos centroids das moscas da imagem

    pred = len(array_predicted)
    gol = len(centroids)
    print('predicted: ',pred)
    print('gold: ',gol)

    for gold in centroids: #valor dos centroids do padrao ouro
        closest = None
        closest_dist = 1000000  # ver maior inteiro
        for predicted in array_predicted:
            dist = abs((np.linalg.norm(gold))-(np.linalg.norm(predicted)))

            if closest == None or dist < closest_dist:
                closest = predicted
                closest_dist = dist

        # predicted que é o mais próximo de gold
        if closest_dist < 20:  # testar esse valor
            verdadeiro_pos = verdadeiro_pos + 1
            output.append('gold {} e predicted {}'.format(gold, closest))
            array_predicted.remove(closest)
        else:
            falso_negativo = falso_negativo + 1
            #print('gold not matched {}'.format(gold))

    falso_positivo = len(array_predicted)
    verdadeiro_neg = 0
       #print('predicted not matched')
    print('Verdadeiro Positivo: ',verdadeiro_pos)
    print('Falso Negativo: ', falso_negativo)
    print('Verdadeiro Negativo: ',verdadeiro_neg)
    print('Falso Positivo: ', falso_positivo)

    #print('saida: ',output)
    #print(predicted)

    # Verdadeiro Positivo = É mosca e o software contou
    # Falso Negativo      = É mosca e o software não contou
    # Verdadeiro Negativo = Não é mosca e o software não contou
    # Falso Positivo      = Não é mosca e o software contou

    with open('saida.csv', 'a', newline='') as saida:

        linha = []
        linha.append(i[0])
        linha.append(pred)
        linha.append(gol)
        linha.append(verdadeiro_pos)
        linha.append(falso_negativo)
        linha.append(verdadeiro_neg)
        linha.append(falso_positivo)

        escrever = csv.writer(saida, delimiter=';')
        escrever.writerow(linha)