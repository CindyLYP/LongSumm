from format import *
import json
import nltk
from nltk.corpus import stopwords
from model.text_rank import rank_scores, gen_embedding
import numpy as np
import random
from tqdm import tqdm

stop_words = stopwords.words('english')


with open("../output/merge/pred_result_1024_preceed.json", 'r', encoding='utf-8') as f:
    d2 = json.load(f)

with open("../output/merge/ex_best.json", 'r', encoding='utf-8') as f:
    d1 = json.load(f)

embedding_path="/home/gitlib/pretrain_model/glove/glove.6B.200d.txt"
emb = gen_embedding(embedding_path)
res = {}
for k in tqdm(d1.keys()):
    s = self_clip(" ".join([d1[k], d2[k]]), r=0.3)
    sents = nltk.sent_tokenize(s)
    words = [len(nltk.word_tokenize(it)) for it in sents]

    scores = rank_scores(sents, emb)
    sents = np.array(sents)
    mask = (scores > 1)
    m_s = " ".join(sents[mask])
    res[k] = m_s
with open("../output/merge/merge_rank.json", 'w') as f:
    json.dump(res, f)
