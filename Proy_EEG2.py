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



# Realizamos una prueba para identificar si está confundido o no
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)

datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['SubjectID','VideoID','Attention','Mediation','Raw','predefinedlabel'],axis=1)
x=datos_EEG.drop(["user-definedlabeln"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["user-definedlabeln"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.05, random_state=25)
# x_entre es la Matriz de entrada para las características
# y_entre es la Matriz de salida
# x_prueba es la matriz de pruebas
# y_prueba es la matriz de salida que supone debería ser, NO es la que se predice
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print("La predicción de si el alumno está confundido o no es de : ")
print(accuracy_score(y_prueba,y_pred))







# Realizamos una prueba para identificar si está confundido o no
from sklearn.linear_model import SGDClassifier
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
x=datos_EEG.drop(["SubjectID"],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG[["SubjectID"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=z, random_state=25)
# x_entre es la Matriz de entrada para las características
# y_entre es la Matriz de salida
# x_prueba es la matriz de pruebas
# y_prueba es la matriz de salida que supone debería ser, NO es la que se predice
clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=100000, tol=1e-3))
clf.fit(x_entre, y_entre.ravel())
y_pred=clf.predict(x_prueba)
print(accuracy_score(y_prueba,y_pred))





!pip install --upgrade tensorflow_hub

import tensorflow_hub as hub

model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = model(["The rain in Spain.", "falls",
"mainly", "In the plain!"])

print(embeddings.shape)  #(4,128)
