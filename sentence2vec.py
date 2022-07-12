import numpy as np
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('ricardo-filho/sbertimbau-large-nli-sts')

def get_vectors(arquivo):
    #sentences = np.load('./preprocessed/trainXnoFilter.npy')
    sentences = np.load(arquivo)
    sentence_embeddings = sbert_model.encode(sentences)
    return sentence_embeddings

TestX = get_vectors('./preprocessed/testX.npy')
TrainX = get_vectors('./preprocessed/trainX.npy')

np.save('preprocessed/testXvec', TestX)
np.save('preprocessed/trainXvec', TrainX)