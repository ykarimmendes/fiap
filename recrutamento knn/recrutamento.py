import pandas as pd

dados = pd.read_excel("cursofiap/recrutamento knn/Recrutamento.xlsx")
#print(dados.shape)
#print(dados.head)

###Agrupa com os dados que tem na coluna
#print(set(dados.ssc_p))

#print(dados.info())

import missingno as msno
import matplotlib.pyplot as plt

###mostra gráficos de nulos
#msno.matrix(dados)
#plt.show()

###Printa o total de nulos
#print(dados.isnull().sum())

import seaborn as sb
import matplotlib.pyplot as plt

###Olha a distribuicao do status em decorrencia do salario
#sb.boxenplot(x="gender", y="salary",data=dados, palette="hls")
#plt.show()

###Preenche coluna de salario nuls com zero
#dados["salary"].fillna(value=0, inplace=True)

###Ver discrepancias dos dados
#sb.boxplot(x=dados["salary"])
#plt.show()

###histogrma (muito importante)
#sb.histplot(data=dados, x="hsc_p")
#plt.show()

###gráfico para ver distribuição
#sb.set_theme(style="whitegrid", palette="muted")
#ax=sb.swarmplot(data=dados, x="mba_p", y="status", hue="workex")
#ax.set(ylabel="mba_p")
#plt.show()

###Gráfico de violino que da para passar o mouse
#import plotly_express as px
#fig=px.violin(dados, y="salary", x="specialisation", color="gender", box=True, points="all" )
#fig.show()

#####----------------------------------------------------------------------------------------------------###################

#####Matriz de correlacao entre todos os dados (MUITO BOM)
from sklearn.preprocessing import LabelEncoder

#Transformar as strings em . Fazer binário em vez de string
colunas=["gender", "workex", "specialisation", "status"]
label_enconder = LabelEncoder()
for col in colunas:
    dados[col] = label_enconder.fit_transform(dados[col])


#Cria estrutura para representar mais variaveis que não sejam apenas duas
#one hot encoding
dummy_hsc_s = pd.get_dummies(dados["hsc_s"], prefix="dummy").astype(int)
dummy_degree_t = pd.get_dummies(dados["degree_t"], prefix="dummy").astype(int)
dados_dummy = pd.concat([dados, dummy_hsc_s ,dummy_degree_t], axis=1)
dados_dummy.drop(["hsc_s", "degree_t", "salary"], axis = 1, inplace=True)
#print(dados_dummy.head())

#Matriz de correlacao
#correlation_matriz = dados_dummy.select_dtypes(include="number").corr().round(2)
#fig, ax = plt.subplots(figsize=(8,8))
#sb.heatmap(correlation_matriz, annot=True, linewidths=0.5, ax=ax)
#plt.show()

#####################

#####Treinar modelo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

x = dados_dummy[["ssc_p", "hsc_p", "degree_p", "workex", "mba_p"]]
y = dados_dummy["status"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=7)
#preprocessamento
#os valores estão muito diferentes, é necessário padronizar as escalas
scaler = StandardScaler()
scaler.fit(x_train)

x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test)

###Muito importante
##Escolher qual valor de K melhor
#após rodar, vivmos que é 5 ou 6. Porém devemos escolher os números impáres

error = []
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_escalonado, y_train)
    pred_i = knn.predict(x_test_escalonado)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12,6))
plt.plot(range(1,10), error, color = "red", linestyle = "dashed", marker = "o", markerfacecolor="blue", markersize=10)
plt.title("Erro médio para knn")
plt.xlabel("Valor de K")
plt.ylabel("Erro médio")
plt.show()

###Verificar a porcentagem de acerto
modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(x_train_escalonado, y_train)
y_predito = modelo_classificador.predict(x_test_escalonado)

#print(accuracy_score(y_test, y_predito))











