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
