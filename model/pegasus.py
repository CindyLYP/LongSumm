import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchsummary import summary
import tensorflow as tf
import tensorflow_datasets as tfds
from utils.build_data import gen_training_data


tokenizer = AutoTokenizer.from_pretrained("./pretrain_model/pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("./pretrain_model/pegasus")


tf.enable_eager_execution()

# d = tfds.load('scientific_papers', data_dir='/data/ysc/tensorflow_datasets/')
# train = d['train']
x, y, sections, summ = gen_training_data()
for i, eg in enumerate(summ):
    gt = summ[i]
    for section in sections[i]:
        if section:
            text = section['text']
            inputs = tokenizer([text], max_length=1024, return_tensors='pt')
            summary_ids = model.generate(inputs['input_ids'])
            pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]
            print("\033[1;31m %s \033[0m" % (section['heading'] if section['heading'] else 'abstract'))
            print(pred)
    print("**"*32)
    print(gt)
    print("--"*64)



print('finish')
