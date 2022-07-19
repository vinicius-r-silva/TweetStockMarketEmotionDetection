import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
# mlp for multi-label classification
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#Abrir arquivos de treino e teste
X_test = np.load('./preprocessed/testXvec.npy')
X_train = np.load('./preprocessed/trainXvec.npy')

Y_test = np.load('./preprocessed/testY.npy')
Y_train = np.load('./preprocessed/trainY.npy')


# get the model
def get_model_MLP(n_inputs, n_outputs):
	model = Sequential()
	#model.add(Dense(512, input_dim=n_inputs, kernel_initializer='random_normal', activation='relu'))
	#model.add(Dropout(0.25))
	model.add(Dense(256, input_dim=n_inputs, kernel_initializer='random_normal', activation='relu'))
	model.add(Dropout(0.25))
	#model.add(Dense(128, input_dim=256, kernel_initializer='random_normal', activation='relu'))
	#model.add(Dropout(0.25))
	model.add(Dense(64, input_dim=256, kernel_initializer='random_normal', activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(n_outputs,input_dim=64, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def get_model_MultiOutput_Logistic():
    return MultiOutputClassifier(LogisticRegression())

def get_model_MultiOutput_SGDC():
    return MultiOutputClassifier(SGDClassifier(max_iter=10000, tol=1e-5,verbose = 0))


def train_model(X, y,model = 'mlp'):
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	callback = EarlyStopping(monitor='loss', patience=10)
	if model == 'mlp':
		model = get_model_MLP(n_inputs,n_outputs)
		model.fit(X_train, y, verbose=0, epochs=1000,callbacks = [callback])
	else:
		if model == 'MultiOutput_Logistic':
			model = get_model_MultiOutput_Logistic()
			model.fit(X_train, y)
		else: 
			model = get_model_MultiOutput_SGDC()
			model.fit(X_train, y)
	return model

def test_model(model,X_test,y_test):
	n_outputs = y_test.shape[1]
	results = model.predict(X_test)
	#creating confusion matrix
	results_table = np.zeros((n_outputs,2))
	#round to nearest integer
	results = np.rint(results)
	for prediction,test in zip(results, y_test):
		for i,(prediction_label,test_label) in enumerate(zip(prediction,test)):
			if(prediction_label == test_label):
				results_table[i][0] += 1
			else:
				results_table[i][1] += 1
	print(results_table)
	return results


emotionLabels = ["TRU","DIS","JOY","SAD","ANT","SUR","ANG","FEA","NEUTRAL"]

model = train_model(X_train,Y_train,'mlp')
results = test_model(model,X_test,Y_test)
print('Acurácia: ', accuracy_score(Y_test, results, normalize=True), '%')
print('hamming score: ', hamming_score(Y_test, results), '%')
print(multilabel_confusion_matrix(Y_test,results))
print(classification_report(Y_test, results, target_names=emotionLabels))

model = train_model(X_train,Y_train,'MultiOutput_Logistic')
results = test_model(model,X_test,Y_test)
print('Acurácia: ', accuracy_score(Y_test, results, normalize=True), '%')
print('hamming score: ', hamming_score(Y_test, results), '%')
print(multilabel_confusion_matrix(Y_test,results))
print(classification_report(Y_test, results, target_names=emotionLabels))

model = train_model(X_train,Y_train,'MultiOutput_SGDC')
results = test_model(model,X_test,Y_test)
print('Acurácia: ', accuracy_score(Y_test, results, normalize=True), '%')
print('hamming score: ', hamming_score(Y_test, results), '%')
print(multilabel_confusion_matrix(Y_test,results))
print(classification_report(Y_test, results, target_names=emotionLabels))

