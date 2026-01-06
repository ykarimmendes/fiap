import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# Etapas do pré-processamento e modelagem

## 1 - Verificar se existem dados nulos (NaN)
# Se existirem, tratar os valores (ex: imputação com 0, média, mediana, etc.)

## 2 - Identificar variáveis categóricas binárias
# Aplicar Label Encoding (ex: gender, workex, status)

## 3 - Identificar variáveis categóricas com mais de duas categorias
# Aplicar One-Hot Encoding (criação de variáveis dummy)

## 4 - Concatenar as variáveis dummy ao dataset original
# Manter apenas colunas numéricas e binárias

## 5 - Remover colunas categóricas originais e colunas irrelevantes
# Ex: colunas de texto, IDs ou variáveis que não serão usadas no modelo

## 6 - Separar variáveis independentes (X) e variável alvo (y)

## 7 - Dividir o dataset em conjunto de treino e teste
# Utilizar stratify quando a variável alvo for desbalanceada

## 8 - Escalonar os dados numéricos
# Etapa obrigatória para algoritmos baseados em distância (KNN)

## 9 - Avaliar diferentes valores de K e treinar o modelo
# Escolher o K que apresenta menor erro ou maior acurácia

## 10 - Avaliar a performance do modelo
# Ex: accuracy, matriz de confusão, precision, recall, F1-score



dados = pd.read_excel("cursofiap/recrutamento knn/Recrutamento.xlsx")

###Preenche coluna de salario nulos com zero
dados["salary"] = dados["salary"].fillna(0)

#Transformar as strings em 0 e 1. Fazer binário em vez de string
colunas=["gender", "workex", "specialisation", "status"]
label_enconder = LabelEncoder()
for col in colunas:
    dados[col] = label_enconder.fit_transform(dados[col])

#Cria estrutura para representar mais variaveis que não sejam apenas duas
#one hot encoding
dummy_hsc_s = pd.get_dummies(dados["hsc_s"], prefix="dummy").astype(int)
dummy_degree_t = pd.get_dummies(dados["degree_t"], prefix="dummy").astype(int)
dados_dummy = pd.concat([dados, dummy_hsc_s ,dummy_degree_t], axis=1)

#Retira colunas que não vamos usar
dados_dummy.drop(["hsc_s", "degree_t", "salary"], axis = 1, inplace=True)

x = dados_dummy[["ssc_p", "hsc_p", "degree_p", "workex", "mba_p"]]
y = dados_dummy["status"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=7)
#preprocessamento
#os valores estão muito diferentes, é necessário padronizar as escalas
scaler = StandardScaler()
scaler.fit(x_train)

x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test)

###Verificar a porcentagem de acerto
#Usar algoritmo para ver qual valor de K é melhor
modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(x_train_escalonado, y_train)
y_predito = modelo_classificador.predict(x_test_escalonado)

print(accuracy_score(y_test, y_predito))

###Outro modelo de predição
#Criar pipeline

svm = Pipeline(
    [
        ("linear_svc", LinearSVC(C=1))
    ]
)

svm.fit(x_train_escalonado, y_train)

y_predito_svm = svm.predict(x_test_escalonado)
print(accuracy_score(y_test, y_predito_svm))