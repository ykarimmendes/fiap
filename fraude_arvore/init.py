import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


dados = pd.read_csv("C:/yehia/dev/cursopython/cursofiap/fraude_arvore/card_transdata.csv")
#print(dados.head())
#print(dados.shape)

##dropando a target
x = dados.drop(columns="fraud") 
##O que quero prever
y = dados["fraud"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=7)

dt = DecisionTreeClassifier(random_state=7,criterion="entropy",max_depth=2)
dt.fit(x_train,y_train)
y_predito=dt.predict(x_test)

class_names = ["Fraude", "Não Fraude"]
label_names = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price", "repeat_retalier", "used_ship", "used_pin_number", "online_order"]
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,15), dpi=300)
tree.plot_tree(dt,feature_names=label_names, class_names=class_names,filled=True)
#fig.savefig("C:/yehia/dev/cursopython/cursofiap/fraude_arvore/arvore_decisao.png")

###Metricas de precisão, revocação, f1-score e acurácia
print(accuracy_score(y_test, y_predito))
