# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:37:22 2020

@author: Larissa
"""

#Identificar imagem de gatos e cachorros COM VALIDACAO CRUZADA

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
import os #pra pegar os nomes das imagens na pasta
import pandas as pd #criar dataframe (arquivo csv) pra guardar as classes
import numpy as np #vetor de zeros pra sortear os indices + testar 1 img
from time import time

#abaixo eh so pra testar 1 img especifica
#from keras.preprocessing import image

start_time = time() #vamos contar quanto tempo demora o codigo

#funcao para ler os arquivos/pastas de dentro de uma pasta
def ler_pasta(endereco):
    arquivo = os.listdir(endereco)
    end_arq =[]
    #print(len(arquivo))
    for i in range(len(arquivo)):
        end_arq.append(endereco+'/'+arquivo[i])
    return end_arq

#ler pastas dentro de animais (cachorro e gato)
endereco_arq = ler_pasta('dataset_animais')

#funcao que cria lista com enderecos e classes das imagens
def endereco_img(i):
    endereco = ler_pasta(endereco_arq[i])
    if endereco_arq[i].split('/')[-1] == 'cachorro':
        classe = 'cachorro'
    else:
        classe = 'gato'
    for j in range(len(endereco)):
           endereco[j] = [endereco[j],classe]
    return endereco

lista_dog = endereco_img(0) #lista dos cachorros
lista_cat = endereco_img(1) #lista dos gatos
lista_animais = lista_dog+lista_cat #lista animais

#cria dataframe com header pra ser usado no flow_from_dataframe
df = pd.DataFrame(lista_animais, columns = ('endereco','classe'))

#agora vamos importar esses dados com flow from dataframe.
##trecho abaixo serve p/ fazer pequenas alteracoes nas imagens pra aumentar o n de dados de treino (augmentation)
gerador_dados = ImageDataGenerator(rescale = 1./255, #normaliza os dados
                                         rotation_range = 7, #grau da rotacao
                                         horizontal_flip = True,  #faz giros horizontais
                                         shear_range = 0.2, #distorce a imagem pra uma direcao
                                         height_shift_range = 0.07, #faixa de mudanca de altura
                                         zoom_range = 0.2   #mexe no zoom
                                         ) #coloca 30% dos dados pra teste

#funcao pra criar o dataframe de treinamento e de teste apartir dos indices do kfold
def criar_df(indice):
    dado = [[]]
    for j in indice:
        dado = dado + [lista_animais[j]]
    dado.pop(0)
    dataf = pd.DataFrame(dado, columns = ('endereco','classe'))
    return dataf

#vamos criar os parametros pra validacao cruzada
seed = 42
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
indices_total = np.zeros(shape = (df.shape[0], 1))
i = 0
resultados = []



for indice_treinamento, indice_teste in kfold.split(df, indices_total):
    print(f'Validacao {i}')
    i += 1
    print('Criando dataframes')
    df_treino = criar_df(indice_treinamento)
    df_teste = criar_df(indice_teste)
    
    
    base_treinamento = gerador_dados.flow_from_dataframe(dataframe = df_treino,
                                      x_col = 'endereco',
                                      y_col = 'classe',
                                      target_size = (64,64),
                                      batch_size = 32,
                                      class_mode = 'binary')
    base_teste = gerador_dados.flow_from_dataframe(dataframe = df_teste,
                                      x_col = 'endereco',
                                      y_col = 'classe',
                                      target_size = (64,64), 
                                      batch_size = 32,
                                      class_mode = 'binary')
    
    ##################rede neural convolucional:#########################
    print('Executando rede')
    classificador = Sequential() #inicia a rede como uma sequencia de camadas
    classificador.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
    #adiciona a camada de convolucao com 32 matrizes de caracteristicas de tamanho 3x3
    #considerando o input imagens de 64x64 pixels e 3 canais de cor RGB com funcao de ativacao relu
    #o input_shape so eh necessario na camada inicial, nas proximas ele ja pega a saida dessa.
    classificador.add(BatchNormalization())   #normalizar os valores das matrizes depois da convolucao, pra melhorar processamento
    classificador.add(MaxPooling2D(pool_size = (2,2))) #pooling com matrizes 2x2
    
    #vamos adicionar uma segunda sequencia de conv+normalizacao+pooling pra melhorar a deteccao de padroes
    classificador.add(Conv2D(32, (3,3), activation = 'relu'))
    classificador.add(BatchNormalization())   
    classificador.add(MaxPooling2D(pool_size = (2,2))) 
    
    #depois de incluidas todas as camadas de conv e pooling por fim damos flatten
    classificador.add(Flatten())
    
    #################rede neural tradicional:################
    n_camadas = 0
    n_neuronios1 = 256
    n_neuronios2 = 256
    drop = 0.2
    epocas = 10
    
    #primeira camada de neuronios
    n_camadas += 1
    classificador.add(Dense(units = n_neuronios1, activation = 'relu')) #a escolha do n de neuronios varia com o tamanho da imagem(numero de entradas)
    classificador.add(Dropout(drop)) #isso faz com que 20% dos neuronios sejam descartados no treinamento e isso diminui a chance de overfitting (varia as caracteristicas q ele observa)
    #segunda camada de neuronios
    n_camadas += 1
    classificador.add(Dense(units = n_neuronios2, activation = 'relu'))
    classificador.add(Dropout(drop))
    #terceira camada de neuronios
    #n_camadas += 1
    #classificador.add(Dense(units = n_neuronios2, activation = 'relu'))
    #classificador.add(Dropout(drop))
    
    #camada de saida da RNA
    classificador.add(Dense(units = 1, activation = 'sigmoid')) #binario (0 ou 1, cachorro ou gato)
    
    #compila a rede neural (esses parametros vou entender melhor nas aulas de RNA tradicional)
    classificador.compile(loss = 'binary_crossentropy', optimizer = 'adam', #loss eh binary em vez de categorial pq so tem 2 opcoes
                          metrics = ['accuracy'])  #metrica pra julgar a qualidade da rede
#The Adam Optimizer is an extension for the Stochastic Gradient Descent which applies ideas from RMSProp 
#and AdaGrad for adapting learning rate during training. 
    
##################rodando a rede pra treinamento e depois pra teste ####################
    classificador.fit(base_treinamento, steps_per_epoch = 4500/32, epochs = epocas) #steps = ((nfolds-1)*5000/(nfolds))/(tamanho batch)
    metricas = classificador.evaluate(base_teste)  #salvar valores de loss e accuracy pra cada rede
    resultados.append(metricas)

df_resul = pd.DataFrame(resultados, columns = ('loss','accuracy'))

end_time = time()
tempo_total = end_time - start_time
    
##################rodando a rede SOMENTE pra treinamento ####################

#classificador.fit_generator(base_treinamento, steps_per_epoch = 4000/32, epochs = 10) 
#4000 eh o total de imagens de treinamento, e 32 eh o batch. Se rodar steps = 4000 fica mais preciso mas demora bem mais

#####################testando uma UNICA imagem###############
##nao esqueca que pra uma unica imagem precisa tratar essa imagem antes de inserir na rede
#imagem_teste = image.load_img('dataset_gatocachorro/test_set/cachorro/dog.3503.jpg',
#                              target_size = (64,64)) #load da img e redimensiona pra 64x64 (padrao da rede)
#imagem_teste = image.img_to_array(imagem_teste)  #converter de img para valores rgb (3 canais)
#imagem_teste /= 255  #normalizar valores da img
#imagem_teste = np.expand_dims(imagem_teste, axis = 0) #deixar no formato de entrada da rede 1x64x64x3 (o 1 eh o numero de valores no batch)
#base_treinamento.class_indices #diz qual a classe 0 e qual classe 1

##rodando de fato o teste
#previsao = classificador.predict(imagem_teste) #predicao da classe dessa img -> gera valor entre 0 e 1.
##previsao eh probabilidade de ser classe 1 (+ proximo de 0 -> classe 0, + proximo de 1 -> classe 1)
#
#if previsao > 0.5:
#    print(f'Essa imagem tem {100*round(previsao[0,0],4)}% de chance de corresponder a um gato')
#else:
#    print(f'Essa imagem tem {100*round(1-previsao[0,0],4)}% de chance de corresponder a um cachorro')