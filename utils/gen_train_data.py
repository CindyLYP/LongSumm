import json
import os
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string/byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, str):
        value = value.encode('unicode-escape')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tfrecords_example(inputs, targets):
    tfrecords_features = {}
    tfrecords_features['inputs'] = _bytes_feature(inputs)
    tfrecords_features['targets'] = _bytes_feature(targets)

    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))


def gen_data_dict():
    def gen_cont(path='./train_json'):
        _, _, files = next(os.walk(path))
        json_list = []
        for file in files:
            with open(path + '/' + file, 'r', encoding='utf-8') as fp:
                raw_d = json.load(fp)
                json_list.append(raw_d)
        return json_list

    def gen_info(path='./abstractive_summaries/by_clusters'):
        info_list = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                filepath = subdir + os.sep + file

                if filepath.endswith(".json"):
                    with open(filepath, 'r') as in_f:
                        summ_info = json.load(in_f)
                        info_list.append(summ_info)

        return info_list

    p1, p2, dt = gen_cont(), gen_info(), []
    for pp1 in p1:
        tmp = {}
        for pp2 in p2:
            name = str(pp2['id']) + '.pdf'
            if pp1['name'] == name:
                tmp['id'] = pp2['id']
                tmp['summary'] = pp2['summary']
                tmp['target'] = " ".join(pp2['summary'])
                tmp['sections'] = pp1['metadata']['sections']
                if tmp['sections']:
                    text = ""
                    for s in tmp['sections']:
                        text += s['text']
                    tmp['body'] = text.replace('\n', ' ')

        dt.append(tmp)
    return dt


def gen_tf_training_data():
    dt = gen_data_dict()

    tfrecord_wrt = tf.python_io.TFRecordWriter('./dataset/train.tfrecord')
    cnt = 0
    for d in dt:
        if 'body' not in d.keys():
            example = get_tfrecords_example(d['body'], d['target'])
            tfrecord_wrt.write(example.SerializeToString())
            cnt += 1
    tfrecord_wrt.close()
    print("total training examples: ", cnt)


def gen_training_data():
    dt = gen_data_dict()
    inp, tar, sec, summ = [], [], [], []
    for d in dt:
        if 'body' in d.keys():
            inp.append(d['body']), tar.append(d['target']), sec.append(d['sections']), summ.append(d['summary'])
    return inp, tar, sec, summ


