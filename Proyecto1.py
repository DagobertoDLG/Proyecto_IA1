import pandas as pd     # Es la forma de importar la librería de pandas
import seaborn as sns   # Es la forma de importar la librería de seaborn
from sklearn.linear_model import LogisticRegression # Importamos la librería de regresión logística de sckitlearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


# Data frame es como una hoja de cálculo compuesto por series, cada serie se pued einterpretar como una columna.
datos_demogr=pd.read_csv("Datos/demographic_info.csv")
datos_EEG=pd.read_csv("Datos/EEG_data.csv")

# Para ver los primeros 10 valores que están en el dataset
datos_demogr.head(10)
datos_EEG.head(10)

# Para ver qué tipo de datos tiene cada una de las columnas en los dos CSV:
datos_demogr.info()
datos_EEG.info()

# Estos que viene es para ver datos estadísticos, percentiles, desviación estándar, cantidad, medias, etc.
datos_demogr.describe()
datos_EEG.describe()

# Para ver los encabezados de las columnas
datos_demogr.columns
datos_EEG.columns

# Para ver los encabezados de las columnas
datos_demogr.columns.tolist()
datos_EEG.columns.tolist()

# Si quiero visualizar únicamente cierta columna:
datos_EEG.Raw
datos_EEG.Raw.head(10)

# O si quiero varias columnas:
datos_EEG[["Raw","Delta"]]
datos_EEG[["Raw","Delta"]].head(10)

# Si quisiéramos quitar algunas columnas utilizamos el comando drop:
datos_EEG_N=datos_EEG.drop(['Raw','Delta'],axis=1)
datos_EEG_N.columns
datos_EEG_N.count()

# Si quiero quitar todo lo que contenga NA, es decir que no tenga valor
datos_EEG_sin_NA=datos_EEG.dropna()
datos_EEG_sin_NA.count()

# Si quiero agrupar por VideoID, además que me dé el promedio por cada uno de los videos
datos_EEG.groupby(["VideoID"]).mean()

# Si quiero ver los datos que hay en la columna de VideoID por ejemplo podemos utilizar lo siguiente:
datos_EEG.VideoID.unique()

# Para agrupar algunos datos pero filtramos únicamente los que tienen VideoID=0
datos_EEG[datos_EEG.VideoID==0].groupby(["user-definedlabeln"]).mean()

# Ahora agrupamos pero por el VideoID y el predefinedlabel, se haría algo así como un diagrama de árbol
datos_EEG.groupby(["VideoID","predefinedlabel"]).mean()
datos_EEG.groupby(["VideoID","predefinedlabel"]).plot() # Este es como el anterior pero graficando, se utiliza la graficadora de numpy

# Aquí utilizamos la graficadora pero de seaborn
ax = sns.boxplot(x="VideoID", y="Attention", hue="predefinedlabel",data=datos_EEG, linewidth=2.5)   # Un diagrama de cajas
sns.histplot(data=datos_EEG, x="Attention",kde=True,binwidth=10)    # Un histograma con una línea que suaviza

# El siguiente comando es para concatenar algunos vectores
df=pd.concat([datos_EEG,datos_EEG])
df.info()

#
datos_EEG.SubjectID.unique()
datos_EEG.VideoID.unique()

# Dummies de SubjectID
Subject_ID=pd.get_dummies(datos_EEG.SubjectID)
Subject_ID=pd.get_dummies(datos_EEG.SubjectID,prefix='SuID')
Subject_ID
# Dummies de VideoID
Video_ID=pd.get_dummies(datos_EEG.VideoID)
Video_ID=pd.get_dummies(datos_EEG.VideoID,prefix='ViID')
Video_ID

# Entonces ahora vamos a poner estas variables en nuestras variables de predicción
x_conca=pd.concat([pd.get_dummies(datos_EEG.SubjectID,prefix='SuID'),pd.get_dummies(datos_EEG.VideoID,prefix='ViID'),datos_EEG.drop(['SubjectID','VideoID'],axis=1)],axis=1)
x_conca
x_conca.info()

# Se elige aleatoriamente quién es el conjunto de entrenamiento y quién es el conjunto de prueba
x=x_conca.drop(['user-definedlabeln'],axis=1)
y=x_conca[["user-definedlabeln"]]
x_entre, x_prueba, y_entre, y_prueba = train_test_split(x.values, y.values, test_size=0.05, random_state=42)
#x_entre.info()
#x_entre
#y_entre.info()
#y_entre
#x_prueba.info()
#y_prueba.info()
#y_entre
# Esto lo comentamos porque la prueba se realizó con el comando sample y repite algunos elementos de prueba que están en el entrenamiento
#x=x_conca.sample(frac=0.9,replace=False,random_state=1)
#x.info()    # Estos son los datos de entrenamiento
#y=x_conca.sample(frac=0.1,replace=False,random_state=5)
#y.info()    # Estos son los datos de pruebas

# Vamos a utilizar libría sckitlearn para hacer la predicción
log_reg=LogisticRegression()
log_reg = LogisticRegression(solver = 'lbfgs',max_iter=10000).fit(x_entre,y_entre.ravel())
y_pred=log_reg.predict(x_prueba)
y_pred
y_prueba
accuracy_score(y_prueba,y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


## Ahora quiero entrar a la parte chida del proyecto. Hay algunos datos que no están en modo dummy, por ejemplo, los videos, el alumno,
## entonces, estos hay que hacerlos dummies.



## Pruebas utilizando muestreo con el comando sample
df = pd.DataFrame({'num_legs': [2, 4, 8, 0],'num_wings': [2, 0, 0, 0],'num_specimen_seen': [10, 2, 1, 8]},index=['falcon', 'dog', 'spider', 'fish'])
df
df.sample(frac=0.9,replace=False,random_state=5)






X, y = np.arange(10).reshape((5, 2)), range(5)
X
list(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train
X_test
y_train
y_test
