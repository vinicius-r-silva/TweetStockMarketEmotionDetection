import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

#Abrir arquivos de treino e teste
X_test = pd.load('./preprocessed/testXvec.npy')
X_train = pd.load('./preprocessed/trainXvec.npy')

Y_test = pd.load('./preprocessed/testY.npy')
Y_train = pd.load('./preprocessed/trainY.npy')

clf = MultiOutputClassifier(LogisticRegression()).fit(X_train, Y_train)

#Teste com uma frase de exemplo
# teste = 'vale5 - vale: excelente noticia Newsletter ADVFN'

print(clf.score(X_test, Y_test))