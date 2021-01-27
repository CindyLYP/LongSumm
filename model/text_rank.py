from utils.gen_train_data import *
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.special import softmax
stop_words = stopwords.words('english')
word_embeddings = {}


def gen_embedding(path):
    word_emb = {}
    with open(path, encoding='utf-8') as f:
        # 按行读取
        for line in tqdm(f):
            values = line.split()
            word_emb[values[0]] = np.asarray(values[1:], dtype='float32')
    return word_emb


def remove_stopwords(s):
    return ' '.join([i for i in s if i not in stop_words])


def  extract_raw_dataset(embedding_path="./dataset/glove_100d.txt"):
    _, _, examples, summary = gen_training_data()
    word_embeddings = gen_embedding(embedding_path)
    train_examples = []
    for example in tqdm(examples):
        train_example = []
        for section in example:

            sentences = sent_tokenize(section['text'])
            if not sentences: continue
            clean_sentences = pd.Series(sentences).str.replace('[^a-zA-Z]', ' ')

            clean_sentences = [remove_stopwords(s.lower().split()) for s in clean_sentences]

            sentences_vectors = []
            for i in clean_sentences:
                # 如果句子长度不为0
                if len(i) != 0:
                    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 1e-2)
                else:
                    v = np.zeros((100,))
                sentences_vectors.append(v)

            similarity_matrix = np.zeros((len(clean_sentences), len(clean_sentences)))

            for i in range(len(clean_sentences)):
                for j in range(len(clean_sentences)):
                    if i != j:
                        similarity_matrix[i][j] = cosine_similarity(
                            sentences_vectors[i].reshape(1, -1), sentences_vectors[j].reshape(1, -1)
                        )

            nx_graph = nx.from_numpy_array(softmax(similarity_matrix, axis=1))
            try:
                scores = nx.pagerank(nx_graph, max_iter=100)
            except:
                print(len(clean_sentences))
                print(clean_sentences)
                continue

            ranked_sentences = sorted(
                ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
            )

            sect = ''
            for rs in ranked_sentences:
                if len(sect)+len(rs[1]) < 900:
                    sect += rs[1]

            train_example.append(sect)
        train_examples.append(train_example)
    return train_examples, summary
