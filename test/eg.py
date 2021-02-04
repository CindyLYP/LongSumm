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
# path = "/data/ysc/pretrain/saved_model"
# imported_model = tf.saved_model.load(path, tags='serve')
# summerize = imported_model.signatures['serving_default']

dataset = tfds.load('scientific_papers/arxiv', data_dir='/data/ysc/tensorflow_datasets',split='test', shuffle_files=False, as_supervised=True)
batch = 16
dataset = dataset.repeat(5).shuffle(1024).batch(batch)

# prefetch 将使输入流水线可以在模型训练时异步获取批处理。
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
aggregator = scoring.BootstrapAggregator()
cnt = 0
for b_ex in dataset:
    cnt += 1
    a = b_ex[0]
    b = b_ex[1]
    for i in range(batch):
        print(len(a[i].numpy().decode('utf-8').split()), len(b[i].numpy().decode('utf-8').split()))
print(cnt)



