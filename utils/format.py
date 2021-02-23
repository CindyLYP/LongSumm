import re
import nltk
from summa import summarizer
from summa import keywords


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