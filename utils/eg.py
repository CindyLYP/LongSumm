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

print("gg")
# tf.enable_eager_execution()
#
# tokenizer = AutoTokenizer.from_pretrained("./pretrain_model/pegasus")
# model = AutoModelForSeq2SeqLM.from_pretrained("./pretrain_model/pegasus")
# print(model)
# d = tfds.load('scientific_papers', data_dir='/data/ysc/tensorflow_datasets/')
# train = d['train']
# for eg in train.take(1):
#     t1, t2, t3 = eg['abstract'], eg['article'], eg['section_names']
#     inputs = tokenizer([t2.numpy().decode('utf-8')], max_length=1024, return_tensors='pt')
#
#     summary_ids = model.generate(inputs['input_ids']
# print("SD")

tf.enable_v2_behavior()
path = "/data/ysc/pretrain/saved_model"
imported_model = tf.saved_model.load(path, tags='serve')
summerize = imported_model.signatures['serving_default']

# dataset = tfds.load('scientific_papers/arxiv', data_dir='/data/ysc/tensorflow_datasets',split='test', shuffle_files=False, as_supervised=True)
#
#
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
aggregator = scoring.BootstrapAggregator()
#
# for ex in tqdm(dataset.take(20), position=0):
#   predicted_summary = summerize(ex[0])['pred_sent'][0]
#   score = scorer.score(ex[1].numpy().decode('utf-8'), predicted_summary.numpy().decode('utf-8'))
#   aggregator.add_scores(score)
# aggregator.aggregate()

with open('../dataset/raw.json', 'r') as f:
    ds = json.load(f)

random.shuffle(ds)
for i, ex in enumerate(ds):
    print("#", end='')
    if i and i % 30 == 0:
        res = aggregator.aggregate()
        print()
        print(res)
        aggregator = scoring.BootstrapAggregator()

    predicted_summary = summerize(tf.convert_to_tensor(ex['body'], dtype=tf.string))['pred_sent'][0]
    score = scorer.score(ex['target'], predicted_summary.numpy().decode('utf-8'))
    aggregator.add_scores(score)
