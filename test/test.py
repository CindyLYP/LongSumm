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
from scripts.eval import rouge_metric
import json
import random
import os


def test_pegasus_model():
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

        predicted_summary = summerize(
            tf.convert_to_tensor(ex['article'], dtype=tf.string))['pred_sent'][0].numpy().decode('utf-8')

        score = scorer.score(ex['summary'], predicted_summary)
        ds[i]['pred'] = predicted_summary
        with open('%d_pred.json' % ds[i]['id'], 'w') as f:
            json.dump(ds[i], f)
        aggregator.add_scores(score)


def test_eval():
    path = "./pegasus"

    info, pred, gt = [], [], []
    for _, _, files in os.walk(path):
        for file in files:
            with open(path+os.sep+file, 'r') as f:
                d = json.load(f)
                pred.append(d['summary'])
                gt.append(d['pred'])
                info.append("paper id: %d, article len: %d, summary len: %d, pred len: %d"
                            % (d['id'], d['article_words'], d['summary_words'], len(d['pred'].split())))
    print("eval data num:", len(gt))
    for i in range(len(gt)):
        print('paper info:', info[i])
        print(rouge_metric([pred[i]], [gt[i]]))
        print('--'*32)
    print("***"*32)
    print('average rouge')
    print(rouge_metric(pred, gt))


def test_roberta_model():
    tf.enable_v2_behavior()
    path = "/data/ysc/pretrain/roberta/saved_model"
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

        predicted_summary = summerize(
            tf.convert_to_tensor(ex['article'], dtype=tf.string))['pred_sent'][0].numpy().decode('utf-8')

        score = scorer.score(ex['summary'], predicted_summary)
        ds[i]['pred'] = predicted_summary
        with open('./roberta/%d_pred.json' % ds[i]['id'], 'w') as f:
            json.dump(ds[i], f)
        aggregator.add_scores(score)


test_roberta_model()