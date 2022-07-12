import pandas as pd
import numpy as np

def changevalues(dataset,column_name):
    dataset.loc[dataset[column_name] < 0, column_name] = 0

def getEmotion(row, emotionLabels):
    res = []
    for x in range(len(row)):
        if(row[x] == 1):
            res.append(emotionLabels[x])

    return res


allTweets = pd.read_csv('./dataset/tweets_stocks.csv')
test = pd.read_csv('./dataset/tweets_stocks-full_agreement.csv')
test_ids = test['tweet_id']

#Separando teste e treino
train = allTweets[~allTweets['tweet_id'].isin(test_ids)]
# print('\n\n train')
# print(train)

#Arrumando valores -2 e -1 para 0 em todas as colunas de emoção
emotion_columns=["TRU","DIS","JOY","SAD","ANT","SUR","ANG","FEA","NEUTRAL"]
for column in emotion_columns:
    changevalues(train, column)
    changevalues(test, column)
    # print(train[column].value_counts())

#Conferindo numero de anotadores de cada dado
print(train['num_annot'].value_counts())
print(test['num_annot'].value_counts())

trainX = []
trainY = []
testX = []
testY = []


#Conferindo quantos dados existentes não possuem nenhuma emoção atribuida
classCount = {}
classCount['no emotion'] = 0
classCount['multiple emotions'] = 0
for emotionLabel in emotion_columns:
    classCount[emotionLabel] = 0

for rowIdx in range(len(test)):
    row = test.iloc[rowIdx]
    rowValues = row[emotion_columns]
    # print('rowValues', type(rowValues))
    emotion = getEmotion(rowValues, emotion_columns)
    
    if(len(emotion) == 0):
        classCount['no emotion'] += 1
        continue
    
    if(len(emotion) > 1):
        classCount['multiple emotions'] += 1
    else:
        classCount[emotion[0]] += 1

    testX.append(row['text'])
    testY.append(rowValues.tolist())
    
print('Test')
print(classCount)

classCount = {}
classCount['no emotion'] = 0
classCount['multiple emotions'] = 0
for emotionLabel in emotion_columns:
    classCount[emotionLabel] = 0

for rowIdx in range(len(train)):
    row = train.iloc[rowIdx]
    rowValues = row[emotion_columns]
    emotion = getEmotion(rowValues, emotion_columns)

    if(len(emotion) == 0):
        classCount['no emotion'] += 1
        continue
    
    if(len(emotion) > 1):
        classCount['multiple emotions'] += 1
    else:
        classCount[emotion[0]] += 1

    trainX.append(row['text'])
    trainY.append(rowValues.tolist())

print('Train')
print(classCount)

trainX = np.array(trainX)
testX = np.array(testX)
trainY = np.array(trainY)
testY = np.array(testY)

print(trainX)
print(testX)
# print(trainY)
# print(testY)

# train.to_csv('./train.csv', index=False)
# test.to_csv('./test.csv', index=False)

np.save('./preprocessed/trainXnoFilter', trainX)
np.save('./preprocessed/trainY', trainY)
np.save('./preprocessed/testXnoFilter', testX)
np.save('./preprocessed/testY', testY)