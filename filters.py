import json
import emoji
from spacy.lang.pt.stop_words import STOP_WORDS
import numpy as np

'''
    Filtros que usaremos:
        - retirar links
        - retirar valores monetários (?)
        - retirar mensões @xxx (?)
    
    Normalizações:
        - acentos
        - maiúsculas
'''

# with open("stopwords.txt") as sw:
#     stopwords = json.load(sw)
# stopwords = stopwords["words"]
stopwords = STOP_WORDS


def filter_symbols(word):
    extras = [',','.',':','\'','"','-','¡','¿','#','?','!','(',')','»','«',';','%','{','}','[',']','$','&','/','=',
              '…','+','-','*','_','^','`','|','°','”','✅','‘','“','⁦','—','⏩','⚠️','✌','➡️','♫','♩','❤','▶','√','🤷‍♀️'
              '🆘','´','`']

    for e in extras:
        word = word.replace(e,'')

    return emoji.demojize(word, delimiters=("", ""))

def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def filter_file(file):
    #stemmer = SnowballStemmer('spanish')

    tweet = [word.lower() for word in file.split()]

    #Filtramos los simbolos
    tweet = [filter_symbols(word) for word in tweet]

    #Filtramos los websites
    tweet = [word for word in tweet if not word.startswith("http") and not word.startswith("@")
             and len(word)]

    #Filtramos los stopwords
    #tweets = [stemmer.stem(word) for word in tweet if word not in stopwords]
    tweets = [normalize(word) for word in tweet if word not in stopwords and not word.isnumeric()]
    
    #filtramos retweets
    if len(tweets)>0 and tweets[0]=="rt":
        tweets.pop(0) 

    return ' '.join(tweets)


setencesTest= np.load('preprocessed/testXnoFilter.npy')
setencesTrain = np.load('preprocessed/trainXnoFilter.npy')

result = []
for setence in setencesTest:
    setence = filter_file(setence)
    result.append(setence)

TestX = np.array(result)


result = []
for setence in setencesTrain:
    setence = filter_file(setence)
    result.append(setence)

TrainX = np.array(result)

print(TrainX)
print(TestX)
np.save('preprocessed/testX', TestX)
np.save('preprocessed/trainX', TrainX)