import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchsummary import summary
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
from rouge_score import rouge_scorer
from rouge_score import scoring
from tqdm import tqdm
import json
import random


tf.enable_v2_behavior()
path = "/data/ysc/pretrain/saved_model"
imported_model = tf.saved_model.load(path, tags='serve')
summerize = imported_model.signatures['serving_default']

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
aggregator = scoring.BootstrapAggregator()

with open('../dataset/train_v1.json', 'r') as f:
    ds = json.load(f)

random.shuffle(ds)
for i, ex in enumerate(ds):
    print("#", end='')
    if i and i % 30 == 0:
        res = aggregator.aggregate()
        print()
        print(res)
        aggregator = scoring.BootstrapAggregator()

    predicted_summary = summerize(tf.convert_to_tensor(ex['article'], dtype=tf.string))['pred_sent'][0]
    score = scorer.score(ex['summary'], predicted_summary.numpy().decode('utf-8'))
    aggregator.add_scores(score)
