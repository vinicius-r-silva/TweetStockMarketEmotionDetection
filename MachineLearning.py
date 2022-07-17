import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, Lasso, SGDClassifier


#Abrir arquivos de treino e teste
X_test = np.load('./preprocessed/testXvec.npy')
X_train = np.load('./preprocessed/trainXvec.npy')

Y_test = np.load('./preprocessed/testY.npy')
Y_train = np.load('./preprocessed/trainY.npy')

print (X_test)
print("\n\n\n\n\n\n")
print(Y_test)
clf = MultiOutputClassifier(SGDClassifier(max_iter=10000, tol=1e-3,verbose = 1 )).fit(X_train, Y_train)

#Teste com uma frase de exemplo
# teste = 'vale5 - vale: excelente noticia Newsletter ADVFN'

print(clf.score(X_test, Y_test))