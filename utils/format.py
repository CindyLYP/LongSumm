import re
import nltk
from summa import summarizer
from summa import keywords
import json
import numpy as np


def drop_sent(raw_s: str):
    sentences = nltk.sent_tokenize(raw_s)
    d_sentences = []
    [d_sentences.append(it) for it in sentences if it not in d_sentences]

    return " ".join(d_sentences)


def add_summ(text):
    res = summarizer.summarize(text)
    return res


def add_keywords(s):
    return keywords.keywords(s)


def sentence_pick():
    with open("../output/test/abt_small.json", 'r', encoding='utf-8') as f:
        d = json.load(f)

    def  sent_sim(s,t):
        ws = set(nltk.word_tokenize(s))
        wt = set(nltk.word_tokenize(t))
        return len(ws & wt) / min(len(ws), len(wt))

    for key in d.keys():
        summary = d[key]
        sents = nltk.sent_tokenize(summary)
        words = [set(nltk.word_tokenize(sent)) for sent in sents]
        mask = np.ones(len(sents))
        for i in range(len(words)):
            k = i
            



sentence_pick()


