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


def sent_sim(s, t, mode='jaccard'):
    if mode == 'jaccard':
        ws = set(nltk.word_tokenize(s))
        wt = set(nltk.word_tokenize(t))
        return len(ws & wt) / len(ws | wt)


def self_clip(raw_str: str, r=0.8):
    sents = nltk.sent_tokenize(raw_str)
    sents = np.array(sents)
    l = len(sents)
    mask = np.ones(l, dtype=bool)
    for i in range(l):
        if not mask[i]:
            continue
        for j in range(i+1, l):
            if sent_sim(sents[i],sents[j], mode='jaccard') >= r:
                mask[j] = False

    selected_sents = sents[mask]
    clip_str = " ".join(selected_sents)
    return clip_str









