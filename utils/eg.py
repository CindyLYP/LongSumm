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
from scripts.eval import rouge_metric

# tf.enable_eager_execution()

tf.enable_v2_behavior()
path = "/data/ysc/pretrain/saved_model"
imported_model = tf.saved_model.load(path, tags='serve')
summerize = imported_model.signatures['serving_default']

dataset = tfds.load('scientific_papers/arxiv', data_dir='/data/ysc/tensorflow_datasets',split='test', shuffle_files=False, as_supervised=True)
batch = 16
dataset = dataset.repeat().shuffle(1024).batch(batch)

# prefetch 将使输入流水线可以在模型训练时异步获取批处理。
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
aggregator = scoring.BootstrapAggregator()
for ex in dataset:
    ext = ex[0]
    predicted_summary = summerize(ex[0])
    predicted_summary = predicted_summary['pred_sent']
    batch_pred, batch_gt = [], []
    for i in range(batch):
        pred = predicted_summary[i].numpy().decode('utf-8')
        gt = ex[1][i].numpy().decode('utf-8')
        batch_pred.append(pred)
        batch_gt.append(gt)
        score = scorer.score(gt, pred)
    for i in range(batch):
        if i%(batch/4) == 0 :
            print()
        else:
            print(len(ex[0].numpy().decode('utf-8').s))

    print(aggregator.aggregate())
    print(rouge_metric(batch_pred, batch_gt))



#
for ex in tqdm(dataset.take(20), position=0):
  predicted_summary = summerize(ex[0])['pred_sent'][0]
  score = scorer.score(ex[1].numpy().decode('utf-8'), predicted_summary.numpy().decode('utf-8'))
  aggregator.add_scores(score)


