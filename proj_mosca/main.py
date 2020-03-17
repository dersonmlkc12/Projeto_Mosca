#!/usr/bin/python
#-*- coding: utf-8 -*-
import cv2
from ler_imagem import ler_imagem
from escreve_imagem import escreve
from configuracao import configuracao
from boi.b_pre_processamento import b_pre_processamento
from boi.b_regiao_interesse import regiao_interesse
from boi.b_watershed import watershed
from boi.b_mascara_bovino import mascara_bovino
from mosca.m_identifica_bordas import identifica_bordas
from mosca.m_melhora_imagem import melhora_imagem
from mosca.m_contornos import regiao_int
from mosca.m_identifica_mosca import identifica_mosca
from parse_labelme_xml import valida

from matplotlib import pyplot as plt
from datetime import datetime
from os import listdir
from os.path import isfile, join
import time

inicio = time.time()

# Atributos
img = ''
data = datetime.now()
fname = data.strftime('%Y-%m-%d-%H-%M-%S')
inputPath = 'image/'
outputPath = 'resultados/'
img = 'image/img1.jpg'

# Constantes
suav_bov = [5, 25, 25]  # Kernel para suavização da imagem do bovino
w_erode = [3,3,1]       # Kernel e repetição para aplicação de erosão do contorno do bovino
w_dilate = [21,21,5]    # Kernel e repetição para aplicação de dilatação do contorno do bovino
pix_cont = [15,70]      # Limiares de pixels de cada borda
perim = [15,45]          # Limiares do perimetro da borda identificada como mosca do chifre

onlyfiles = [f for f in listdir(inputPath) if isfile(join(inputPath, f))]

#for n in range(0, len(onlyfiles)):
    #img = inputPath + onlyfiles[n]

# Leitura da Imagem
original = ler_imagem(img)
imagem = ler_imagem(img)

# Pre processamento
pre = b_pre_processamento(imagem,suav_bov)

# Definição da região de interesse
regiao = regiao_interesse(pre)

# Algoritmo Watershed para segmentar o bovino
water = watershed(regiao, imagem, w_erode, w_dilate)

# Extrai a mascara do bovino da imagem original
mascara = mascara_bovino(water[0], water[1], imagem)

# Detecta as bordas da imagem
bordas = identifica_bordas(mascara)

# Filtros de melhoramento na imagem
melhora = melhora_imagem(bordas[1])

# Limpa contornos do bovino
contornos = regiao_int(melhora[0],original,pix_cont)

# Identifica as moscas-do-chifre e realiza a contagem
ident = identifica_mosca(contornos, imagem, perim)
total = ident[1]
resultad = escreve(ident[0], total)

valida(ident[3],img)
# Imprime as configurações utilizadas na imagem
config = ['Imagem: '+str(img), 'Resolucao: '+str(original.shape), 'Suavizacao Bov: GaussianBlur '+str(suav_bov),
          'Dilatacao: '+str(w_dilate),'Erosao: '+str(w_erode),
          'Pixels do contorno: '+str(pix_cont),'Perimetro das bordas: '+str(perim)]

config = configuracao(config)

# Converte imagens para RGB
original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
mascara = cv2.cvtColor(mascara,cv2.COLOR_BGR2RGB)
resultado = cv2.cvtColor(resultad,cv2.COLOR_BGR2RGB)
teste1 = cv2.cvtColor(ident[2],cv2.COLOR_BGR2RGB)

titles = ['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)']
images = [original, pre, regiao, water[0], mascara, bordas[0], bordas[1], melhora[0], teste1, resultado]

for i in range(10):
    plt.subplot(3,4,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i],fontsize=8)
    plt.xticks([]),plt.yticks([])

plt.savefig(outputPath+fname+'.jpg', dpi=1200)
cv2.imwrite(outputPath + 'res_' + fname + '.jpg', resultad)

#plt.savefig(outputPath+onlyfiles[n]+'_fases.jpg', dpi=1200)
#cv2.imwrite(outputPath + 'res_' + onlyfiles[n] + '.jpg', resultad)
print('Salvo ', img+' total: ', total)
#plt.show()
#fim = time.time()
#print('duracao: %f'%(fim-inicio))

#key = cv2.waitKey(0)
#if key == 27:
    #cv2.destroyAllWindows()