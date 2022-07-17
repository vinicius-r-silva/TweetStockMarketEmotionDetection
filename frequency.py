import numpy as np

def getEmotion(row, emotionLabels):
    res = []
    for x in range(len(row)):
        if(row[x] == 1):
            res.append(emotionLabels[x])

    return res

def addCounter(matrix, emotions, word):
    for emotion in emotions:
        if(emotion not in matrix):
            matrix[emotion] = {}

        if(word not in matrix[emotion]):
            matrix[emotion][word] = 0

        matrix[emotion][word] += 1

emotionLabels = ["TRU","DIS","JOY","SAD","ANT","SUR","ANG","FEA","NEUTRAL"]
X_test = np.load('./preprocessed/testX.npy')
X_train = np.load('./preprocessed/trainX.npy')

Y_test = np.load('./preprocessed/testY.npy')
Y_train = np.load('./preprocessed/trainY.npy')

TermEmotionMatrix = {}
for rowIdx in range(len(X_train)):
    words = X_train[rowIdx].split()
    emotions = getEmotion(Y_train[rowIdx], emotionLabels)
    for word in words:
        addCounter(TermEmotionMatrix, emotions, word)

print(TermEmotionMatrix)