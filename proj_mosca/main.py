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
from mosca.m_identifica_mosca import identifica_mosca

from mosca.m_contornos import regiao_int
from mosca.m_watershed import m_watershed
from mosca.conta_mosca_bkp import conta_mosca
from matplotlib import pyplot as plt
from datetime import datetime

# Atributos
img = 'image/img16.jpg'
data = datetime.now()
fname = data.strftime('%Y-%m-%d-%H-%M-%S')
suav_bov = [7,7]
w_erode = [3,3,1]
w_dilate = [7,7,10]
m_sobelx = [1,0,3,0.3]
m_sobely = [0,1,3,0.3]

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

#m_watershed(mascara)
#conta_mosca(mascara,original)

# Detecta as bordas do bovino
bordas = identifica_bordas(mascara,m_sobelx,m_sobely)

# Filtros de melhoramento na imagem
melhora = melhora_imagem(bordas)

# Limpa contornos do bovino
contornos = regiao_int(melhora)

# Identifica as moscas-do-chifre e realiza a contagem
m_watershed(contornos,original)
ident = identifica_mosca(contornos, imagem)
total = ident[1]
resultado = escreve(ident[0], total)

# Imprime as configurações utilizadas na imagem
config = ['Imagem: '+str(img), 'Suavizacao Bov: GaussianBlur '+str(suav_bov),"Dilatacao: "+str(w_dilate), "Erosao: "+str(w_erode),
          "Sobel x: "+str(m_sobelx), "Sobel y: "+str(m_sobely)]
configuracao = configuracao(config)


titles = ['Original','Pre Processamento','Regiao Interesse','Watershed','Mascara','Bordas','Imagem Melhorada','Contornos','Resultado','Configuração']
images = [original, pre, regiao, water[0], mascara, bordas, melhora,contornos, resultado,configuracao]

for i in range(10):
    plt.subplot(3,4,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i],fontsize=8)
    plt.xticks([]),plt.yticks([])

#plt.savefig('resultados/'+fname+'.jpg', dpi=600)
#plt.show()

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()