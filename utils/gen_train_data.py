import json
import os
import tensorflow as tf


def gen_cont(path='./train_json'):
    _, _, files = next(os.walk(path))
    json_list = []
    for file in files:
        with open(path+'/'+file, 'r', encoding='utf-8') as fp:
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

p1 = gen_cont()
p2 = gen_info()
dt = []
for pp1 in p1:
    tmp = {}
    for pp2 in p2:
        name = str(pp2['id'])+'.pdf'
        if pp1['name'] == name:
            tmp['id'] = pp2['id']
            tmp['summary'] = pp2['summary']
            tmp['sections'] = pp1['metadata']['sections']
            try:
                text = ""
                for s in tmp['sections']:
                    text += s['text']

                tmp['body'] = text.replace('\n', ' ')
            except:
                pass
    dt.append(tmp)


def _bytes_feature(value):
    """Returns a bytes_list from a string/byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tfrecords_example(inputs, targets):
    tfrecords_features = {}
    feat_shape = inputs.shape
    tfrecords_features['inputs'] = tf.train.Feature(bytes_list=
                                                     tf.train.BytesList(value=[inputs.tostring()]))
    tfrecords_features['targets'] = tf.train.Feature(bytes_list=
                                                     tf.train.BytesList(value=[targets.tostring()]))

    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))


# 创建tfrecord的writer，文件名为xxx
tfrecord_wrt = tf.python_io.TFRecordWriter('demo.tfrecord')
# 把数据写入Example
exmp = get_tfrecords_example(feats[inx], labels[inx])
# Example序列化
exmp_serial = exmp.SerializeToString()
# 写入tfrecord文件
tfrecord_wrt.write(exmp_serial)
# 写完后关闭tfrecord的writer
tfrecord_wrt.close()

d = ''
