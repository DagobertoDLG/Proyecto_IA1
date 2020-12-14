import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # Matriz de entrada para las características
X
type(X)
y = np.array([1, 2, 5, 7])  # Matriz de salida para las salidas
y
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)
print(clf.predict([[-1.6, -0.8]]))  #Representa la matriz de prueba, tiene características





from sklearn import svm
X = [[0, 0], [1, 1]]
X
y = [0, 1]
y
clf = svm.SVC()
clf.fit(X, y)
SVC()
clf.predict([[2., 2.]])




from sklearn import svm
X = [[5, 3], [2, 8]]
y = [4, 7]
clf = svm.SVC()
clf.fit(X, y)
SVC()
clf.predict([[5., 8.]])

for x in range(0, 19):
    z=round(1-x*0.05-0.05,2)
    print(z)
    #print(type(z))

x=np.array([[1,2],[3,4]])
x
x[0,1]
print(x[:,1])

z=[0.05:0.05:0.95]

z=np.zeros(5)
for x in range(0, 5):
    print("We're on time %d" % (x))
    z[x]=np.array(x*2)
z

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
X
Y = np.array([1, 1, 2, 2])
Y
# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))



## Aquí vamos a aprender embeddings
from numpy import array
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
indices
distances


import numpy as np
from sklearn.neighbors import NearestNeighbors
samples = [[1,2,3], [0,0,0], [-2,3,1], [-1,-1,-1]]
neigh = NearestNeighbors(n_neighbors=2, radius=10)
neigh.fit(samples)
neigh.kneighbors([[-5,-5,-2]], 2, return_distance=True)
#nbrs = neigh.radius_neighbors([[-100,-100,-2]], return_distance=False)
nbrs = neigh.radius_neighbors([[-5,-5,-2]], 1000, return_distance=False)
np.asarray(nbrs[0][0])
xx
yy


import numpy as np
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
samples
neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
neigh.fit(samples)
neigh.kneighbors([[0, 1, 1.3]], 2, return_distance=False)
nbrs = neigh.radius_neighbors([[0, 0, 3.1]], 4, return_distance=False)
np.asarray(nbrs[0][0])

a=np.array([[1,2],[3,4],[-5,7]])
a
a[1,:]




import statistics as stat
edades = [21, 17, 89, 76, 32, 21, 45, 21, 89, 21, 15, 89, 21]
moda=stat.mode(edades)
print(moda)





from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.1, random_state=42)
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)
print(nca_pipe.score(X_test, y_test))
X
X_test
y_test
nca_pipe

toto




import statistics as stat
edades=[1,2,1,2,1,2,1,2,3,4,5]
edades = [21, 17, 89, 76, 32, 21, 45, 21, 89, 21, 15, 89, 21]
moda=stat.mode(edades)
print(moda)
mode(edades)
type(edades)


moda(a[3][:])
print(y_entre[indice[0]])
z=y_entre[indice[0]]
zz=z.tolist()
moda(zz)
moda((y_entre[indice[0]]).tolist())









from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.arange(6).reshape(3, 2)
X
poly = PolynomialFeatures(degree=2)
poly.fit_transform(X)




from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])
# fit to an order-3 polynomial data
x = np.arange(5)
y = 3 - 2 * x + x ** 2 - x ** 3
model = model.fit(x[:, np.newaxis], y)
model.named_steps['linear'].coef_




# Aprendiendo perceptrón
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
poly.fit_transform(X)
y = X[:, 0]^ X[:, 1]   #Operación XOR
X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
X
clf = Perceptron(fit_intercept=False, max_iter=10, tol=None,shuffle=False).fit(X, y)
clf.predict(X)
clf.score(X, y)



















# Aprendiendo red neuronal
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

clf.predict([[2., 2.], [-1., -2.]])

[coef.shape for coef in clf.coefs_]


clf.predict_proba([[2., 2.], [1., 2.]])


X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)
clf.predict([[1., 2.]])
clf.predict([[0., 0.]])









from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
X_train
X_scaled



















from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
y=print(scaler.transform(data))
y
print(scaler.transform([[2, 2]]))












import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
class_names
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

confusion_matrix(X_test, y_test, labels=["setosa", "versicolor", "virginica"])

class_names

for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()









from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
(tn, fp, fn, tp)

disp = plot_confusion_matrix(classifier,y_true, y_pred,cmap=plt.cm.Blues,display_labels=['ant', 'bird', 'cat'],normalize=None)






from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
y_pred=clf.predict([[-0.5,-0.5]])
y_pred







from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

clf.predict([[2., 2.]])














import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

X, y = datasets.load_iris(return_X_y=True)
#X.shape, y.shape    ## La forma de una matriz es el número de elementos en cada dimensión

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
#X_train.shape, y_train.shape
#X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.model_selection import ShuffleSplit
#n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)










import numpy as np
from sklearn.model_selection import ShuffleSplit
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
y = np.array([1, 2, 1, 2, 1, 2])
X
y

rs = ShuffleSplit(n_splits=5, test_size=0.33, random_state=0)
rs
rs.get_n_splits(X)
print(rs)

for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

rs = ShuffleSplit(n_splits=4, train_size=0.66, test_size=.16,random_state=0)
for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)










# Análisis de componentes principales PCA
# Reducción de dimensionalidad
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
pca.transform(X)

pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
pca.transform(X)

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
pca.transform(X)

import numpy as np
from sklearn.decomposition import IncrementalPCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
ipca = IncrementalPCA(n_components=2, batch_size=3)
ipca.fit(X)
ipca.transform(X) # doctest: +SKIP


import numpy as np
from sklearn.decomposition import PCA
X = np.array([[1,0.5,8],[2,0.9,7],[3,0.5,-5],[4,0,4],[5,-0.5,10],[6,-0.9,-3],[7,0.5,-7]])
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
pca.transform(X)







# Insertar una columna en un dataframe
# Import pandas package
import pandas as pd

# Define a dictionary containing Students data
data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Height': [5.1, 6.2, 5.1, 5.2],
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)
df
# Declare a list that is to be converted into a column
address = ['Delhi', 'Bangalore', 'Chennai', 'Patna']

# Using 'Address' as the column name
# and equating it to the list
df['Address'] = address

# Observe the result
df









# Insertar en dataframe pero de forma condicional
import pandas
import pandas as pd

df = pd.DataFrame( {'TheValue':['4','6','9','7'] , 'Date':
['02/20/2019','01/15/2019','08/21/2019','02/02/2019']})
df['Date']=pd.to_datetime(df.Date)
df.sort_values(by='Date')

df['Result']="Pass"

df["Results"] = df.apply(lambda x: "Pass" if int(x["TheValue"]) > 5 else "Fail", axis=1)
print(df)













# Insertar en dataframe pero de forma condicional
import pandas
import pandas as pd

df = pd.DataFrame( {'TheValue':['4','8','10','9'] , 'Date':
['02/20/2019','01/15/2019','08/21/2019','02/02/2019']})
df['Date']=pd.to_datetime(df.Date)

competencias=pd.DataFrame({'Calificación':['8','9','10'],'Competencia':['SA','DE','AU']})
competencias

def pass_or_fail(row):
    result = "Fail"
    if int(row["TheValue"]) >= 10:
        result = "AU"
    elif  int(row["TheValue"]) >= 9:
        result = "DE"
    elif  int(row["TheValue"]) >= 8:
        result = "SA"
    else:
        result = "NA"
    return result


def pass_or_fail(row):
    result = "Fail"
    if int(row["TheValue"]) >= 10:
        result = "AU"
    elif  int(row["TheValue"]) >= 9:
        result = "DE"
    elif  int(row["TheValue"]) >= 8:
        result = "SA"
    else:
        result = "NA"
    return result

df["Results"] = df.apply(pass_or_fail, axis=1)
print(df)




if condición_1:
    bloque 1
elif condición_2:
    bloque 2
else:
    bloque 3
