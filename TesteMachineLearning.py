import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from scipy import spatial

X, y = make_multilabel_classification(n_classes=3, random_state=0)
clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)

query = "I need to learn NLP"

# sentences = [
#     "This is an awesome book to learn NLP.",
#     "DistilBERT is an amazing NLP model.",
#     "We can interchangeably use embedding, encoding, or vectorizing.",
# ]

sentences = np.load('./preprocessed/trainX.npy')
sbert_model = SentenceTransformer('ricardo-filho/sbertimbau-large-nli-sts')
sentence_embeddings = sbert_model.encode(sentences)
query_vec = sbert_model.encode([query])[0]
for sent in sentences:
  sim = spatial.distance.cosine(query_vec, sbert_model.encode([sent])[0])
  print("Sentence = ", sent, "; similarity = ", sim)
