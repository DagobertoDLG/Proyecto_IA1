# Proyecto señales encefalográficas
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def moda(datos):
    repeticiones = 0
    for i in datos:
        n=datos.count(i)
        if n>repeticiones:
            repeticiones = n
    moda =[] #Arreglo donde se guardara el o los valores de mayor frecuencia
    #moda=np.array([])
    for i in datos:
        n = datos.count(i) # Devuelve el número de veces que x aparece enla lista.
        if n == repeticiones and i not in moda:
            moda.append(i)
    if len(moda) != len(datos):
        return(moda)
    else:
        return([0])

        # Elección del método
# Random forest             RF
# Máquina de vectores:      SVC
# K-nearest neighnbors:     KNN
# KNN varios vecinos:       KNNM
# Red neuronal:             MLP
# Decision tree:            DT
# Descenso de gradiente:    SGD

        # Tipo de escalamiento
# Ninguno:                  NONE
# Standarscaler:            STD

Modelo="RF"     # Elección del clasificador
Escala="NONE"   # Elección del escalamiento
ts=0.1          # Tamaño del test size

# Leemos CSV donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
# Fusionamos ambos CSV de manera que se tenga una tabla total sobre el género, edad, persona, etc.
datos_demogr.rename(columns={'subject ID': 'SubjectID'}, inplace=True)
datos_EEG = datos_demogr.merge(datos_EEG, on='SubjectID')
datos_EEG.rename(columns={' age': 'age', ' ethnicity': 'ethnicity', ' gender': 'gender'}, inplace=True)
datos_EEG['Label'] = datos_EEG['user-definedlabeln'].astype(np.int)
datos_EEG['gender'] = datos_EEG['gender'].apply(lambda x: 1  if x == 'M' else 0)
ethnicity_dummies = pd.get_dummies(datos_EEG['ethnicity'])
datos_EEG = pd.concat([datos_EEG, ethnicity_dummies], axis=1)
datos_EEG = datos_EEG.drop('ethnicity', axis=1)
datos_EEG=datos_EEG.drop(['predefinedlabel','user-definedlabeln'],axis=1)
#datos_EEG.head(0)                           # Estos son los títulos del data frame
x=datos_EEG.drop(["SubjectID"],axis=1)      # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]                  # Salida: sujeto

# Se realiza el escalamiento
print("Tipo de escalamiento: ")
if Escala=="NONE":
    print("Ninguno")
    x=x.values
elif Escala=="STD":
    print("StandardScaler")
    scaler = StandardScaler()
    scaler.fit(x)
    x=scaler.transform(x)
else:
    print("No eligió")
    x=x.values

x_entre, x_prueba, y_entre, y_prueba = train_test_split(x, y.values, test_size=ts, random_state=25)
# Se realiza la predicción
if Modelo=="RF":
    print("Se eligió como modelo: Random forest")
    clf = RandomForestClassifier(n_estimators=250)
    clf = clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x, y.values.ravel(), cv=cv)
    print("Media de validación cruzada: ",np.mean(Validacion_cruzada))
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(confusion_matrix(y_prueba, y_pred))

    cm = confusion_matrix(y_prueba, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False)
    plt.xlabel("Predecida")
    plt.ylabel("Real")
    plt.title("Matriz de confusión")
    plt.show()
elif Modelo=="SVC":
    print("Se eligió como modelo: SVC")
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf=clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x, y.values.ravel(), cv=cv)
    print("Media de validación cruzada: ",np.mean(Validacion_cruzada))
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="KNN":
    print("Se eligió como modelo:K-Nearest neighbors")
    samples=x_entre
    neigh = NearestNeighbors(n_neighbors=1, radius=5000000)
    neigh.fit(samples)
    indice=neigh.kneighbors(x_prueba, 1, return_distance=False)
    y_entre=np.ravel(y_entre)
    y_pred=np.zeros(len(indice))
    for i in range(0,len(indice)):
        y_pred[i]=y_entre[indice[i]]
    accuracy=accuracy_score(y_prueba,y_pred)
    print("El accuracy score es de: ",accuracy)
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="KNNM":
    print("Se eligió como modelo:K-Nearest neighbors múltiple")
    samples=x_entre
    neigh = NearestNeighbors(n_neighbors=9, radius=5000000)
    neigh.fit(samples)
    indice=neigh.kneighbors(x_prueba, 9, return_distance=False)
    y_entre=np.ravel(y_entre)
    y_pred=np.zeros([len(indice)])
    for i in range(0,len(indice)):
        y_pred[i]=moda((y_entre[indice[i]]).tolist())[0]
    accuracy=accuracy_score(y_prueba,y_pred)
    print("El accuracy score es de: ",accuracy)
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="MLP":
    print("Se eligió como modelo:Perceptrón multicapa")
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,3,2,5), random_state=1,max_iter=10000)# Este da 30.76%
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=1,max_iter=10000) # Este da 46%
    clf=clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x, y.values.ravel(), cv=cv)
    print("Media de validación cruzada: ",np.mean(Validacion_cruzada))
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="DT":
    print("Se eligió como modelo:Decision tree")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x, y.values.ravel(), cv=cv)
    print("Media de validación cruzada: ",np.mean(Validacion_cruzada))
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="SGD":
    print("Se eligió como modelo:Descneso del gradiente estocástico")
    clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000, tol=1e-3))
    clf=clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x, y.values.ravel(), cv=cv)
    print("Media de validación cruzada: ",np.mean(Validacion_cruzada))
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(confusion_matrix(y_prueba, y_pred))
else:
    print("No eligió bien, sin embargo, elegimos random forest")
    clf = RandomForestClassifier(n_estimators=250)
    clf = clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x, y.values.ravel(), cv=cv)
    print("Media de validación cruzada: ",np.mean(Validacion_cruzada))
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(confusion_matrix(y_prueba, y_pred))
