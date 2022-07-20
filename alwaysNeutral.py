import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.svm import SVC


def TFIDF(X_train, X_test):
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test set and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5)

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer.vocabulary_

def train_classifier(X_train, y_train, C, regularisation):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    model = OneVsRestClassifier(LogisticRegression(penalty=regularisation, C=C, max_iter=10000, multi_class='multinomial', solver='saga')).fit(X_train, y_train)
    #model=OneVsRestClassifier(SVC()).fit(X_train, y_train)
    return model

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def getEmotion(row, emotionLabels):
    res = []
    for x in range(len(row)):
        if(row[x] == 1):
            res.append(emotionLabels[x])

    return res

def addCounter(matrix, emotions, word, allEmotions):
    for emotion in allEmotions:
        if(word not in matrix[emotion]):
            matrix[emotion][word] = 0

    for emotion in emotions:
        matrix[emotion][word] += 1

emotionLabels = ["TRU","DIS","JOY","SAD","ANT","SUR","ANG","FEA","NEUTRAL"]
X_test = np.load('./preprocessed/testX.npy')
X_train = np.load('./preprocessed/trainX.npy')

Y_test = np.load('./preprocessed/testY.npy')
Y_train = np.load('./preprocessed/trainY.npy')


X_train_tfidf, X_test_tfidf, tfidf_vocab = TFIDF(X_train, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
print(tfidf_reversed_vocab)

#Treinamento
classifier_tfidf = train_classifier(X_train_tfidf, Y_train, C = 10, regularisation = 'l2')

Y_predicted = classifier_tfidf.predict(X_test_tfidf)
Y_predicted = np.zeros((334,9), dtype=int)
Y_predicted[:,8] = 1
# print(Y_predicted)
# print(type(Y_predicted))
# print(len(Y_predicted))
# print(Y_predicted.dtype)
# print(Y_predicted.shape)
# print(Y_predicted.dtype)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_test_tfidf)

print('Acurácia: ', 100*accuracy_score(Y_test, Y_predicted, normalize=True), '%')
print('Hamming score: {0}%'.format(100*hamming_score(Y_test, Y_predicted)))
print(classification_report(Y_test, Y_predicted, target_names=emotionLabels, zero_division=0))

print(multilabel_confusion_matrix(Y_test, Y_predicted))