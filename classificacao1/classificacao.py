#import pandas as pd
#import matplotlib.pyplot as plt


#dados = pd.read_excel("cursofiap/classificacao1/gaf_esp.xlsx")
###vê os dados que estão na base
#print(dados.head()) 
#print(dados.describe()) descreve os minimos, maximos, e etc da base

###Mostra a qantidade de linhas e colunas
#print(dados.shape)

###Agrupa por alguma coluna
#print(dados.groupby("Espécie").describe())

###gráfico de dispersão para saber como estão os dados
#dados.plot.scatter(x="Comprimento do Abdômen", y="Comprimento das Antenas")
#plt.show()

from sklearn import *
import pandas as pd
from sklearn.model_selection import train_test_split

dados = pd.read_excel("cursofiap/classificacao1/gaf_esp.xlsx")
x = dados[["Comprimento do Abdômen", "Comprimento das Antenas"]]
###target
y = dados["Espécie"]
###Mostra as espécies e mostra a quantidade
#print(y.value_counts())

###Equlibrio da base de teste e treinamento. Pode ter mais de um tipo e menos de outro e para não inviezar algoritmo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state = 42)




