import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from confussion_matrix import get_confussion_matrixes

def TFIDF(X_train, X_test):
    """
        X_train, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test set and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, token_pattern='(\S+)')

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer.vocabulary_

def train_classifier(X_train, y_train, C, regularisation):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    model = OneVsRestClassifier(LogisticRegression(penalty=regularisation, C=C, max_iter=10000)).fit(X_train, y_train)
    return model

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
classifier_tfidf = train_classifier(X_train_tfidf, Y_train, C = 4, regularisation = 'l2')

y_predicted_labels_tfidf = classifier_tfidf.predict(X_test_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_test_tfidf)

print('Acurácia: ', accuracy_score(Y_test, y_predicted_labels_tfidf, normalize=True), '%')
print(get_confussion_matrixes(Y_test,y_predicted_labels_tfidf))

# TF_matrix = {}
# for emotion in emotionLabels:
#     TF_matrix[emotion] = {}

# for rowIdx in range(len(X_train)):
#     words = X_train[rowIdx].split()
#     emotions = getEmotion(Y_train[rowIdx], emotionLabels)
#     for word in words:
#         addCounter(TF_matrix, emotions, word)

# # TF_matrix['JOY']['termo']
# termEmotionMatrix = {}

# #IDF = log10(N/df(t))
# for emotion in TF_matrix:
#     for word in TF_matrix[emotion]:
#         termEmotionMatrix[word] = {}
#         for emotion in TF_matrix:
#             termEmotionMatrix[word][emotion] = TF_matrix[emotion][word]

# IDF_matrix = {}

# for word in termEmotionMatrix:
#     IDF=0
#     for emotion in termEmotionMatrix[word]:
#         if termEmotionMatrix[word][emotion]!=0:
#             IDF+=1
    
#     IDF_matrix[word]=np.log(len(emotionLabels)/IDF)

# for emotion in TF_matrix:
#     for word in TF_matrix[emotion]:
#         TF_matrix[emotion][word] += 1
#         TF_matrix[emotion][word] = math.log10(TF_matrix[emotion][word])



#print(TF_matrix)