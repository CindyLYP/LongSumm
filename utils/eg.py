import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchsummary import summary
import tensorflow as tf
import tensorflow_datasets as tfds
tf.enable_eager_execution()

tokenizer = AutoTokenizer.from_pretrained("./pretrain_model/pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("./pretrain_model/pegasus")
print(model)
d = tfds.load('scientific_papers', data_dir='/data/ysc/tensorflow_datasets/')
train = d['train']
for eg in train.take(1):
    t1, t2, t3 = eg['abstract'], eg['article'], eg['section_names']
    inputs = tokenizer([t2.numpy().decode('utf-8')], max_length=1024, return_tensors='pt')

    summary_ids = model.generate(inputs['input_ids'])

print("SD")

