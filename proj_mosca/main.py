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
from mosca.m_water import waterb
from mosca.m_identifica_mosca import identifica_mosca

from matplotlib import pyplot as plt
from datetime import datetime

# Atributos
img = 'image/img16.jpg'
data = datetime.now()
fname = data.strftime('%Y-%m-%d-%H-%M-%S')

suav_bov = [7,7]        # Kernel para suavização da imagem do bovino
w_erode = [2,2,1]       # Kernel e repetição para aplicação de erosão do contorno do bovino
w_dilate = [22,22,6]    # Kernel e repetição para aplicação de dilatação do contorno do bovino
m_bordas = [60, 120]    # Limiares para extração de bordas das moscas
pix_cont = [20,500]     # Limiares de pixels de cada borda
perim = [10,60]         # Limiares do perimetro da borda identificada como mosca do chifre

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
bordas = identifica_bordas(mascara,m_bordas)

# Filtros de melhoramento na imagem
melhora = melhora_imagem(bordas)

# Limpa contornos do bovino
contornos = regiao_int(melhora,original,pix_cont)

# Identifica as moscas-do-chifre e realiza a contagem
labelm = waterb(contornos)
ident = identifica_mosca(labelm, contornos, imagem, perim)
total = ident[1]
resultado = escreve(ident[0], total)

# Imprime as configurações utilizadas na imagem
config = ['Imagem: '+str(img), 'Resolucao: '+str(original.shape), 'Suavizacao Bov: GaussianBlur '+str(suav_bov),"Dilatacao: "+str(w_dilate),
          "Erosao: "+str(w_erode),"Bordas: "+str(m_bordas),"Pixels do contorno: "+str(pix_cont),
          "Perimetro das bordas: "+str(perim)]
configuracao = configuracao(config)

# Converte imagens para RGB
original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
mascara = cv2.cvtColor(mascara,cv2.COLOR_BGR2RGB)
resultado = cv2.cvtColor(resultado,cv2.COLOR_BGR2RGB)

titles = ['Original','Pre Processamento','Regiao Interesse','Watershed','Mascara','Bordas','Imagem Melhorada','Contornos','Resultado','Configuração']
images = [original, pre, regiao, water[0], mascara, bordas, melhora, contornos, resultado, configuracao]


for i in range(10):
    plt.subplot(3,4,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i],fontsize=8)
    plt.xticks([]),plt.yticks([])

plt.savefig('resultados/'+fname+'.jpg', dpi=1200)
plt.show()

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()