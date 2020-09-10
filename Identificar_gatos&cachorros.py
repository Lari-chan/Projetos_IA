# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:46:33 2020

@author: Larissa
"""

#Identificar imagem de gatos e cachorros

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#abaixo eh so pra testar 1 img especifica
import numpy as np
from keras.preprocessing import image

##################rede neural convolucional:#########################

classificador = Sequential() #inicia a rede como uma sequencia de camadas
classificador.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
#adiciona a camada de convolucao com 32 matrizes de caracteristicas de tamanho 3x3
#considerando o input imagens de 28x28 pixels e 3 canais de cor RGB com funcao de ativacao relu
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

classificador.compile(loss = 'binary_crossentropy', optimizer = 'adam', #loss eh binary em vez de categorial pq so tem 2 opcoes
                      metrics = ['accuracy'])  

#trecho abaixo serve p/ fazer pequenas alteracoes nas imagens pra aumentar o n de dados de treino (augmentation)
gerador_treinamento = ImageDataGenerator(rescale = 1./255, #normaliza os dados
                                         rotation_range = 7, #grau da rotacao
                                         horizontal_flip = True,  #faz giros horizontais
                                         shear_range = 0.2, #distorce a imagem pra uma direcao
                                         height_shift_range = 0.07, #faixa de mudanca de altura
                                         zoom_range = 0.2   #mexe no zoom
                                         )
gerador_teste = ImageDataGenerator(rescale = 1./255) #no teste so precisa normalizar

#o codigo abaixo puxa os dados dentro das pastas do seu pc, sendo a pasta dataset contendo 
#o training_set (com as pastas 'gato' e 'cachorro') e o test_set (com os mesmos nomes de pasta dentro)
#ele ja ira identificar e avisar quantas imagens e quantas classes foram encontradas (nesse caso 2 classes)
base_treinamento = gerador_treinamento.flow_from_directory('dataset_gatocachorro/training_set',
                                                           target_size = (64,64), #igual ao tamanho informado no conv2D
                                                           batch_size = 32,
                                                           class_mode = 'binary') #binario pq sao 2 classes
base_teste = gerador_treinamento.flow_from_directory('dataset_gatocachorro/test_set',
                                                           target_size = (64,64), 
                                                           batch_size = 32,
                                                           class_mode = 'binary')

##################rodando a rede SOMENTE pra treinamento ####################
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000/32, epochs = 10) 
#4000 eh o total de imagens de treinamento, e 32 eh o batch. Se rodar steps = 4000 fica mais preciso mas demora bem mais

####################testando uma UNICA imagem###############
#nao esqueca que pra uma unica imagem precisa tratar essa imagem antes de inserir na rede
imagem_teste = image.load_img('dataset_gatocachorro/test_set/cachorro/dog.3503.jpg',
                              target_size = (64,64)) #load da img e redimensiona pra 64x64 (padrao da rede)
imagem_teste = image.img_to_array(imagem_teste)  #converter de img para valores rgb (3 canais)
imagem_teste /= 255  #normalizar valores da img
imagem_teste = np.expand_dims(imagem_teste, axis = 0) #deixar no formato de entrada da rede 1x64x64x3 (o 1 eh o numero de valores no batch)
base_treinamento.class_indices #diz qual a classe 0 e qual classe 1

#rodando de fato o teste
previsao = classificador.predict(imagem_teste) #predicao da classe dessa img -> gera valor entre 0 e 1.
#previsao eh probabilidade de ser classe 1 (+ proximo de 0 -> classe 0, + proximo de 1 -> classe 1)

if previsao > 0.5:
    print(f'Essa imagem tem {100*round(previsao[0,0],4)}% de chance de corresponder a um gato')
else:
    print(f'Essa imagem tem {100*round(1-previsao[0,0],4)}% de chance de corresponder a um cachorro')





###se todas essas imagens tivessem sem separar treino e teste, mas dentro de pastas 'gato' e 'cachorro'
#    
#gerador_dados = ImageDataGenerator(rescale = 1./255, 
#                                   rotation_range = 7, 
#                                   horizontal_flip = True,
#                                   shear_range = 0.2, 
#                                   height_shift_range = 0.07, 
#                                   zoom_range = 0.2,  
#                                   validation_split=0.3)  #separa 30% pra teste
#
##cria dados de treinamento e dados de teste
#base_treinamento = gerador_dados.flow_from_directory('dataset_animais',
#                                                           target_size = (64,64), 
#                                                           batch_size = 32,
#                                                           class_mode = 'binary',
#                                                           subset='training',seed=42) 
#base_teste = gerador_dados.flow_from_directory('dataset_animais',
#                                                           target_size = (64,64), 
#                                                           batch_size = 32,
#                                                           class_mode = 'binary',
#                                                           subset='validation',seed=42)

