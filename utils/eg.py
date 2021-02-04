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

