import tensorflow_datasets as tfds
import tensorflow as tf
import json
tf.enable_v2_behavior()


path = './dataset/train_v2.json'
with open(path, 'r') as f:
    d = json.load(f)
raw_d = {}
raw_d['document'] = []
raw_d['summary'] = []
for ex in d:
    if len(ex['summary'].split())< 100: continue
    raw_d['document'].append(ex['article'])
    raw_d['summary'].append(ex['summary'])
with open('./dataset/bigbird_train.json', 'w') as f:
    json.dump(raw_d,f)

ds = tf.data.Dataset.from_tensor_slices(raw_d)
print(len(raw_d['summary']))
batch = 16
dataset = ds.repeat(5).shuffle(1024).batch(batch)
cnt = 1
for b_ex in dataset:
    cnt += 1
    a = b_ex['document']
    b = b_ex['summary']
    for i in range(batch):
        print(len(a[i].numpy().decode('utf-8').split()), len(b[i].numpy().decode('utf-8').split()))
