# Projetos_IA
### Repositório para guardar meus projetos de IA/ML/DP


- Identificar_gatos&cachorros.py -> Identificador de gatos e cachorros:

Código em python de rede neural convolucional para identificar gatos e cachorros. Os dados estão em pastas "treino" e "teste" e dentro de cada uma destas existem pastas "cachorro" e "gato" com as respectivas imagens.


- Identificar_animais_dadosjuntos.py -> Identificador de gatos e cachorros sem divisão entre treino/test e uso de validação cruzada/Kfold:

Código em python de rede neural convolucional com validação cruzada para identificar gatos e cachorros que utiliza kfold (k = 10) para dividir entre dados de treino e teste. Os dados estão em pastas "cachorro" e "gato" com as respectivas imagens. As métricas (loss e accuracy) dos 10 treinos da validação cruzada são salvos em um dataframe.

- Predicting_churn.ipynb -> Predicting voluntary churn in a telecommunication company:

Código em python comentado em inglês que visa a prever se um dado consumidor de uma empresa de telecomunicações irá abandonar o contrato com base em seus dados (classificação através de aprendizado supervisionado). Nesse desafio é realizada a limpeza e análise exploratória dos dados dos consumidores, seleção de melhores features/variáveis para previsão de abandono (churn), teste com 3 modelos distintos de Machine learning (Random forest, K-nearest neighbors/KNN e Rede neural artificial/RNA) e por fim avaliação desses modelos e escolha do melhor utilizando diversas métricas. Os resultados das previsões para os clientes teste de todos os 3 métodos se encontram em "Churn_PREDICTIONS.csv".
