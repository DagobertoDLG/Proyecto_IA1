# Proyecto señales encefalográficas
import pandas as pd
import numpy as np
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
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn import tree

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
# Escalador                 ESC

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
print("Tipo de escalamiento: Ninguno")
if Escala="NONE"
    print("Ninguno")
    x=x.values
elif Escala="STD":
    print("StandardScaler")
    scaler = StandardScaler()
    scaler.fit(x)
    x=scaler.transform(x)
elif Escala="ESC":
    print("scale")
    x=preprocessing.scale(x)
else:
    print("No eligió")
    x=x.values

x_entre, x_prueba, y_entre, y_prueba = train_test_split(x, y.values, test_size=ts, random_state=25)
# Se realiza la predicción
if Modelo=="RF":
    clf = RandomForestClassifier(n_estimators=250)
    clf = clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x.values, y.values.ravel(), cv=cv)
    print("Validación cruzada: ",Validaciones_cruzada)
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(accuracy_score(y_pred,y_prueba))
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="SVC":
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf=clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x.values, y.values.ravel(), cv=cv)
    print("Validación cruzada: ",Validaciones_cruzada)
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(accuracy_score(y_pred,y_prueba))
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="KNN":
    samples=x_entre
    neigh = NearestNeighbors(n_neighbors=k, radius=5000000)
    neigh.fit(samples)
    indice=neigh.kneighbors(x_prueba, 1, return_distance=False)
    y_entre=np.ravel(y_entre)
    y_pred=np.zeros(len(indice))
    for i in range(0,len(indice)):
        y_pred[i]=y_entre[indice[i]]
    print(accuracy_score(y_prueba,y_pred))
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="KNNM":
    samples=x_entre
    neigh = NearestNeighbors(n_neighbors=9, radius=5000000)
    neigh.fit(samples)
    indice=neigh.kneighbors(x_prueba, k, return_distance=False)
    y_entre=np.ravel(y_entre)
    y_pred=np.zeros([len(indice)])
    for i in range(0,len(indice)):
        y_pred[i]=moda((y_entre[indice[i]]).tolist())[0]
    print(accuracy_score(y_prueba,y_pred))
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="MLP":
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,3,2,5), random_state=1,max_iter=10000)# Este da 30.76%
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=1,max_iter=10000) # Este da 46%
    clf=clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x.values, y.values.ravel(), cv=cv)
    print("Validación cruzada: ",Validaciones_cruzada)
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(accuracy_score(y_pred,y_prueba))
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="DT":
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x.values, y.values.ravel(), cv=cv)
    print("Validación cruzada: ",Validaciones_cruzada)
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(accuracy_score(y_pred,y_prueba))
    print(confusion_matrix(y_prueba, y_pred))
elif Modelo=="SGD":
    clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000, tol=1e-3))
    clf=clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x.values, y.values.ravel(), cv=cv)
    print("Validación cruzada: ",Validaciones_cruzada)
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(accuracy_score(y_pred,y_prueba))
    print(confusion_matrix(y_prueba, y_pred))
else:
    clf = RandomForestClassifier(n_estimators=250)
    clf = clf.fit(x_entre, y_entre.ravel())
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    Validacion_cruzada=cross_val_score(clf, x.values, y.values.ravel(), cv=cv)
    print("Validación cruzada: ",Validaciones_cruzada)
    # La matriz de confusión es
    y_pred=clf.predict(x_prueba)
    print(accuracy_score(y_pred,y_prueba))
    print(confusion_matrix(y_prueba, y_pred))
























# Análisis con todos los datos
# Random forest
import pandas as pd
import numpy as np
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
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_demogr.rename(columns={'subject ID': 'SubjectID'}, inplace=True)
datos_EEG = datos_demogr.merge(datos_EEG, on='SubjectID')
datos_EEG.rename(columns={' age': 'age', ' ethnicity': 'ethnicity', ' gender': 'gender'}, inplace=True)
datos_EEG['Label'] = datos_EEG['user-definedlabeln'].astype(np.int)
datos_EEG['gender'] = datos_EEG['gender'].apply(lambda x: 1  if x == 'M' else 0)
ethnicity_dummies = pd.get_dummies(datos_EEG['ethnicity'])
datos_EEG = pd.concat([datos_EEG, ethnicity_dummies], axis=1)
datos_EEG = datos_EEG.drop('ethnicity', axis=1)
datos_EEG=datos_EEG.drop(['predefinedlabel','user-definedlabeln'],axis=1)

x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto

#x=preprocessing.scale(x.values)
#scaler = StandardScaler()
#scaler.fit(x)
#x=scaler.transform(x)

pca = PCA(n_components='mle', svd_solver='full')
pca=pca.fit(x.values)
x=pca.transform(x.values)
x
x.shape

x_entre, x_prueba, y_entre, y_prueba = train_test_split(x, y.values, test_size=0.1, random_state=25)
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
cross_val_score(clf, x, y.values.ravel(), cv=cv)
# Sin PCA
#array([0.58190328, 0.56552262, 0.56864275, 0.58658346, 0.56396256])
np.mean([0.91575663, 0.90327613, 0.91887676, 0.90873635, 0.89937598])
np.mean([0.9149766 , 0.90171607, 0.91263651, 0.90327613, 0.90093604])
np.mean([0.90717629, 0.89469579, 0.90015601, 0.89079563, 0.88143526])















# Análisis con componentes principales
# Random forest
import pandas as pd
import numpy as np
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
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto

#x=preprocessing.scale(x.values)
#scaler = StandardScaler()
#scaler.fit(x.values)
#x=scaler.transform(x)

pca = PCA(n_components='mle', svd_solver='full')
pca=pca.fit(x.values)
x=pca.transform(x.values)

x_entre, x_prueba, y_entre, y_prueba = train_test_split(x, y.values, test_size=0.1, random_state=25)
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
cross_val_score(clf, x, y.values.ravel(), cv=cv)

# Con PCA sin escalamiento
#array([0.50624025, 0.48907956, 0.49765991, 0.50936037, 0.49687988])
np.mean([0.50624025, 0.48907956, 0.49765991, 0.50936037, 0.49687988])














# Análisis de componentes principales sin Raw, Meditation ni Attention
# Random forest
import pandas as pd
import numpy as np
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
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto

#x=preprocessing.scale(x.values)
#scaler = StandardScaler()
#scaler.fit(x.values)
#x=scaler.transform(x)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(x.values)
x=pca.transform(x.values)

x_entre, x_prueba, y_entre, y_prueba = train_test_split(x, y.values, test_size=0.1, random_state=25)
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
cross_val_score(clf, x, y.values.ravel(), cv=cv)
# Sin PCA
#array([0.50468019, 0.46567863, 0.48439938, 0.50468019, 0.475039  ])
np.mean([0.50468019, 0.46567863, 0.48439938, 0.50468019, 0.475039  ])

# Con PCA sin escalamiento
#array([0.3798752 , 0.37441498, 0.39079563, 0.39625585, 0.37675507])
np.mean([0.3798752 , 0.37441498, 0.39079563, 0.39625585, 0.37675507])

# Con PCA escalamiento processing.scale


# Con PCA escalamiento StandardScaler

















































# Realizamos una prueba para identificar si está confundido o no
import pandas as pd     # Es la forma de importar la librería de pandas
import seaborn as sns   # Es la forma de importar la librería de seaborn
from sklearn.linear_model import LogisticRegression # Importamos la librería de regresión logística de sckitlearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['SubjectID','VideoID','Attention','Mediation','Raw','predefinedlabel'],axis=1)
x=datos_EEG.drop(["user-definedlabeln"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["user-definedlabeln"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_prueba,y_pred))
confusion_matrix(y_prueba, y_pred)





# Realizamos una prueba para identificar si está confundido o no
import pandas as pd     # Es la forma de importar la librería de pandas
import seaborn as sns   # Es la forma de importar la librería de seaborn
from sklearn.linear_model import LogisticRegression # Importamos la librería de regresión logística de sckitlearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['SubjectID','VideoID','Attention','Mediation','Raw','predefinedlabel'],axis=1)
x=datos_EEG.drop(["user-definedlabeln"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["user-definedlabeln"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_entre, y_entre.ravel())
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
cross_val_score(clf, x.values, y.values.ravel(), cv=cv)

y_pred=clf.predict(x_prueba)
print(accuracy_score(y_prueba,y_pred))
confusion_matrix(y_prueba, y_pred)















import pandas as pd     # Es la forma de importar la librería de pandas
import seaborn as sns   # Es la forma de importar la librería de seaborn
from sklearn.linear_model import LogisticRegression # Importamos la librería de regresión logística de sckitlearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
datos_demogr.head(10)
datos_EEG.head(10)
# Para ver qué tipo de datos tiene cada una de las columnas en los dos CSV:
datos_demogr.info()
datos_EEG.info()
# Estos que viene es para ver datos estadísticos, percentiles, desviación estándar, cantidad, medias, etc.
datos_demogr.describe()
datos_EEG.describe()

# Realizamos la prueba para ver si coincide el sujeto
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
zz=np.zeros(19)
t=np.zeros(19)
for i in range(0, 19):
    z=round(1-i*0.05-0.05,2)
    x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=z, random_state=25)
    # x_entre es la Matriz de entrada para las características
    # y_entre es la Matriz de salida
    # x_prueba es la matriz de pruebas
    # y_prueba es la matriz de salida que supone debería ser, NO es la que se predice
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_entre, y_entre.ravel())
    y_pred=clf.predict(x_prueba)
    zz[i]=accuracy_score(y_prueba,y_pred)
    t[i]=z;
plt.plot(t,zz)
plt.title("Accuracy vs Tamaño de la prueba")
plt.xlabel("Accuracy")
plt.ylabel("Tamaño de la prueba")
plt.show()

























# Realizamos una prueba para un solo valor de test_size
# Modelo descenso del gradiente estocástico, support vector machines
import pandas as pd     # Es la forma de importar la librería de pandas
import seaborn as sns   # Es la forma de importar la librería de seaborn
from sklearn.linear_model import LogisticRegression # Importamos la librería de regresión logística de sckitlearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
# x_entre es la Matriz de entrada para las características
# y_entre es la Matriz de salida
# x_prueba es la matriz de pruebas
# y_prueba es la matriz de salida que supone debería ser, NO es la que se predice
clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=10000, tol=1e-3))
clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_prueba,y_pred))
confusion_matrix(y_prueba, y_pred)

cv = ShuffleSplit(n_splits=5, test_size=0.01, random_state=0)
cross_val_score(clf, x.values, y.values.ravel(), cv=cv)
















# K-Nearest neighbors con un solo vecino
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
# La cantidad de vecinos cercanos son:
k=1
samples=x_entre
neigh = NearestNeighbors(n_neighbors=k, radius=5000000)
neigh.fit(samples)
indice=neigh.kneighbors(x_prueba, k, return_distance=False)
#indice=np.ravel(indice)
y_entre=np.ravel(y_entre)
y_pred=np.zeros(len(indice))

for i in range(0,len(indice)):
    y_pred[i]=y_entre[indice[i]]
print(accuracy_score(y_prueba,y_pred))
confusion_matrix(y_prueba, y_pred)





















# K-Nearest neighbors con varios vecinos
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
# La cantidad de vecinos cercanos son:
k=9
samples=x_entre
neigh = NearestNeighbors(n_neighbors=k, radius=5000000)
neigh.fit(samples)
indice=neigh.kneighbors(x_prueba, k, return_distance=False)
y_entre=np.ravel(y_entre)
y_pred=np.zeros([len(indice)])
for i in range(0,len(indice)):
    #print(y_entre[indice[i]])
    #print(moda((y_entre[indice[i]]).tolist())[0])
    y_pred[i]=moda((y_entre[indice[i]]).tolist())[0]
print(accuracy_score(y_prueba,y_pred))
confusion_matrix(y_prueba, y_pred)







# Ahora utilizamos una red neuronal artificial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
#x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values,stratify=y.values, test_size=0.001, random_state=42)
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)

x=preprocessing.scale(x.values)
x_entre=preprocessing.scale(x_entre)
x_prueba=preprocessing.scale(x_prueba)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,3,2,5), random_state=1,max_iter=1000000)# Este da 30.76%
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,10,10,5), random_state=1,max_iter=10000)
clf=clf.fit(x_entre, y_entre.ravel())

y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
cross_val_score(clf, x, y.values.ravel(), cv=cv)


















# Ahora utilizamos una red neuronal artificial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
#x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values,stratify=y.values, test_size=0.001, random_state=42)
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)

scaler = StandardScaler()
scaler.fit(x.values)
x=scaler.transform(x.values)

scaler = StandardScaler()
scaler.fit(x_entre)
x_entre=scaler.transform(x_entre)
scaler.fit(x_prueba)
x_prueba=scaler.transform(x_prueba)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,3,2,5), random_state=1,max_iter=10000)# Este da 30.76%
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=1,max_iter=10000) # Este da 46%
clf=clf.fit(x_entre, y_entre.ravel())

y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
cross_val_score(clf, x, y.values.ravel(), cv=cv)
















# Random forest
import pandas as pd
import numpy as np
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

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
cross_val_score(clf, x.values, y.values.ravel(), cv=cv)






















# Random forest con standarscaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
scaler = StandardScaler()
scaler.fit(x_entre)
x_entre=scaler.transform(x_entre)
scaler.fit(x_prueba)
x_prueba=scaler.transform(x_prueba)
x_prueba
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)




























# Random forest con scale
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
x_entre=preprocessing.scale(x_entre)
x_prueba=preprocessing.scale(x_prueba)
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)


























# Árbol de decisión
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_demogr
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.1, random_state=25)
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_pred,y_prueba))
confusion_matrix(y_prueba, y_pred)
