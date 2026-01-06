import pandas as pd

#plot de gráficos
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn  as sns

#algorito de Agrupamento
from sklearn.cluster import KMeans, DBSCAN

#Avaliação de desempenho
from sklearn.metrics import adjusted_rand_score, silhouette_score

annual_imcome = "Annual Income (k$)"
spending_score = "Spending Score (1-100)"
gender = "Gender"

def plot_dbscan_com_outlier():
    ###Plotando com outliers
    plt.scatter(dados[[annual_imcome]], dados[[spending_score]],c=dbscan_labels, alpha=0.5, cmap="rainbow")
    plt.title(("DBScan - Com outlier"))
    plt.xlabel("Salário anual")
    plt.ylabel("Pontuação de gastos")
    plt.show()

def plot_dbscan_sem_outlier():
    ###Plotando sem outliers
    mascara = dbscan_labels>=0
    plt.scatter(dados[[annual_imcome]][mascara], dados[[spending_score]][mascara],c=dbscan_labels[mascara], alpha=0.5, cmap="rainbow")
    plt.title(("DBScan - Sem outlier"))
    plt.xlabel("Salário anual")
    plt.ylabel("Pontuação de gastos")
    plt.show()

def plot_clusters_escalonados():
    plt.figure(figsize=(10,5))

    plt.scatter(
        dados_escalonados[annual_imcome],
        dados_escalonados[spending_score],
        c=kmeans_labels_escalonados,
        cmap="rainbow",
        alpha=0.5
    )

    plt.scatter(
        centroides_escalonados[:, 0],
        centroides_escalonados[:, 1],
        c="black",
        marker="X",
        s=200,
        alpha=0.5
    )

    plt.xlabel("Salário anual (escalonado)")
    plt.ylabel("Pontuação de gastos (escalonado)")
    plt.show()

##### Passo 1 #####
### Importar dados
dados = pd.read_csv("cursofiap/cluster_shopping/mall.csv")
#print(dados.shape)
#print(dados.head())

##### Passo 2 #####
## Verificar se tem nulos
#print(dados.isnull().sum())

##### Passe 3 #####
## Análise Exploratória
## Análise geral
#print(dados.describe())

## Análise de campo específico
#print(dados[annual_imcome].median())

## Análise de distribuição de variáveis
#dados.hist(figsize=(12,12))
#plt.show()

## Análise de correlação entre variáveis
#plt.figure(figsize=(6,4))
#sns.heatmap(dados[['Age', annual_imcome, spending_score]].corr(method='pearson'),annot=True, fmt=".1f");
#plt.show()

##analisando proporções entre gêneros
#print(dados[gender].value_counts())

##Gráfico para analisar o género em relação as variáveis. 
##Entender possíveis grupos. Observar o X
#sns.pairplot(dados, hue=gender)
#plt.show()

#### Parte 4 #####
## Feature Scaling
## Padronização ou normalização dos dados

from sklearn.preprocessing import StandardScaler, MinMaxScaler

##Escolher o escalonamento de acordo com os dados
scaler = StandardScaler() #Escalona entre -1 e 1
#scaler = MinMaxScaler() #Escalona entre 0 e 1
scaler.fit(dados[[annual_imcome,spending_score]])
dados_escalonados = scaler.transform(dados[[annual_imcome, spending_score]])

#### Parte 5 #####
##Aplicando Kmeans - Pipeline
##Chutamos o n_clusters = 6. Mas não é o melhor. no passo 7 vamos ver como fazer para encontrar o melhor n

##Definindo modelo de clusterização. K-means
kmeans = KMeans(n_clusters=6, random_state=0)
##Treinando
kmeans_escalonados = kmeans.fit(dados_escalonados)

##Salvando os centroides de cada cluster
centroides_escalonados = kmeans.cluster_centers_

##Salvanos labels que queremos encontrar
kmeans_labels_escalonados = kmeans.predict(dados_escalonados)
dados_escalonados = pd.DataFrame(dados_escalonados, columns=[annual_imcome, spending_score])
dados_escalonados["Grupos"] = kmeans_labels_escalonados

#plot_clusters_escalonados()

#### Parte 7 #####
##Lista com a quantidade de clusters que iremos testar
#sse=[]
#k = list(range(1,10))
#for i in k:
#    kmeans = KMeans(n_clusters=i, random_state=0)
#    kmeans.fit(dados[[annual_imcome, spending_score]])
##    #calculo de erros do kmeans
#    sse.append(kmeans.inertia_) 

#plt.rcParams['figure.figsize'] = (10,5)
#plt.plot(k, sse, "-o")
#plt.xlabel(r"Numero de cluster")
#plt.ylabel("Inércia")
##Vimos que o melhor K é 5
#plt.show()

###Analisando dados
##Para olhar os grupos criados, pode fazer um groupby
#print(dados_escalonados.groupby("Grupos")[annual_imcome].mean())

####Fazendo treinamento sem normalizacao
#Vamos usar 5 K pq encontramos que é o melhor no passo anterior
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(dados[[annual_imcome, spending_score]])
centroides = kmeans.cluster_centers_
kmeans_labels = kmeans.predict(dados[[annual_imcome, spending_score]])

##Plotando
#plt.scatter(dados[[annual_imcome,]], dados[[spending_score]], c=kmeans_labels, alpha=0.5, cmap="rainbow")
#plt.xlabel("Salário anual")
#plt.ylabel("Pontuação de gastos")
#plt.scatter(centroides[:,0], centroides[:,1], c='black', marker="X", s=200, alpha=0.5)
#plt.rcParams["figure.figsize"] = (10,5)
#plt.show()

##olhando os grupos
#dados['Grupos'] = kmeans_labels
#dados_grupo_1 = dados[dados["Grupos"]==0]
#print(dados_grupo_1[annual_imcome].mean())
#dados_grupo_2 = dados[dados["Grupos"]==1]
#print(dados_grupo_2[annual_imcome].mean())
#dados_grupo_3 = dados[dados["Grupos"]==2]
#print(dados_grupo_3[annual_imcome].mean())
#dados_grupo_4 = dados[dados["Grupos"]==3]
#print(dados_grupo_4[annual_imcome].mean())
#dados_grupo_5 = dados[dados["Grupos"]==4]
#print(dados_grupo_5[annual_imcome].mean())

################DBScan - Tira ruidos
dbscan = DBSCAN(eps=10, min_samples=8)
dbscan.fit(dados[[annual_imcome,spending_score]])
dbscan_labels = dbscan.labels_
#print(dbscan_labels)
##Labels com -1 são considerados outlairs

#plot_dbscan_sem_outlier()
#plot_dbscan_com_outlier

###############Avaliar os desempenhos
print(adjusted_rand_score(kmeans_labels, dbscan_labels)) #Resultado de 0.71 = bom
#Se tiver perto de 1 está em melhor segregração
print(silhouette_score(dados[[annual_imcome, spending_score]],kmeans_labels))
print(silhouette_score(dados[[annual_imcome, spending_score]],dbscan_labels))





