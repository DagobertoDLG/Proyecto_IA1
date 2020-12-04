import pandas as pd     # Es la forma de importar la librería de pandas
import seaborn as sns   # Es la forma de importar la librería de seaborn
from sklearn.linear_model import LogisticRegression # Importamos la librería de regresión logística de sckitlearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

# Leemos donde provienen los datos
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")
datos_EEG=datos_EEG.drop(['VideoID','Attention','Mediation','Raw','predefinedlabel','user-definedlabeln'],axis=1)
# Para ver los primeros 10 valores que están en el dataset
datos_demogr.head(10)
datos_EEG.head(10)
# Para ver qué tipo de datos tiene cada una de las columnas en los dos CSV:
datos_demogr.info()
datos_EEG.info()
# Estos que viene es para ver datos estadísticos, percentiles, desviación estándar, cantidad, medias, etc.
datos_demogr.describe()
datos_EEG.describe()

# Por el momento vamos a trabajar con las señales Delta ... Gamma  para reconocer qué persona es
ID=pd.get_dummies(datos_EEG.SubjectID,prefix='ID')
datos_EEG=pd.concat([ID,datos_EEG.drop(['SubjectID'],axis=1)],axis=1)
# Primero hacemos la prueba con el sujeto 0
datos_EEG0=datos_EEG.drop(['ID_1.0','ID_2.0','ID_3.0','ID_4.0','ID_5.0','ID_6.0','ID_7.0','ID_8.0','ID_9.0'],axis=1)
# Realizamos la prueba
x=datos_EEG0.drop(['ID_0.0'],axis=1)    # Entrada no considera el sujeto, solo sus características
y=datos_EEG0[["ID_0.0"]]    # Salida, quién es el sujeto
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.05, random_state=42)
#log_reg=LogisticRegression()
log_reg = LogisticRegression(solver = 'lbfgs',max_iter=10000).fit(x_entre,y_entre.ravel())
#log_reg = LogisticRegression(solver = 'lbfgs',max_iter=10000).fit(x_entre,y_entre)
y_pred=log_reg.predict(x_prueba)
accuracy_score(y_prueba,y_pred)
len(y_prueba)
len(y_pred)
y_prueba
Prueba=pd.concat([y_prueba,y_pred],axis=0)
datos_EEG.groupby(["VideoID"]).mean()

a=np.array([1,2,5,8])
a=np.transpose(a)
a
b=np.array([2,3,8,10])
b=np.transpose(b)
b
Prueba=pd.concat([a,b])
