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
from matplotlib import pyplot as plt
from datetime import datetime

# Atributos
img = 'image/img4.jpg'
data = datetime.now()
fname = data.strftime('%Y-%m-%d-%H-%M-%S')
suav_bov = [3,3]
w_erode = [3,3,2]
w_dilate = [5,5,10]

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

# Detecta as bordas do bovino
bordas = identifica_bordas(mascara)

# Filtros de melhoramento na imagem
melhora = melhora_imagem(bordas)

# Identifica as moscas-do-chifre e realiza a contagem
ident = identifica_mosca(melhora, imagem)
total = ident[1]
resultado = escreve(ident[0], total)

# Imprime as configurações utilizadas na imagem
config = ['Imagem: '+str(img), 'Suavizacao Bov: GaussianBlur '+str(suav_bov),"Dilatacao: "+str(w_dilate), "Erosao: "+str(w_erode)]
configuracao = configuracao(config)


titles = ['Original','Pre Processamento','Regiao interesse','Watershed','Mascara','Bordas','Imagem Melhorada','Resultado','Configuração']
images = [original, pre, regiao, water[0], mascara, bordas, melhora, resultado,configuracao]

for i in range(9):
    plt.subplot(3,3,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i],fontsize=8)
    plt.xticks([]),plt.yticks([])

plt.savefig('resultados/'+fname+'.jpg', dpi=600)
plt.show()

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()