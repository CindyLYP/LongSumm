import json
import os
# from xml.dom.minidom import parse
from xml.etree.ElementTree import parse
from lxml import etree
from xml.etree.ElementTree import iterparse
import pandas as pd
import re


import tensorflow as tf


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
        tmp['body'] = ""
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
        if not tmp['body'] or not tmp['target']:
            continue
        dt.append(tmp)

    with open('./dataset/raw.json', 'w') as f:
        json.dump(dt, f)
    return dt


def gen_tf_training_data():
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

    dt = gen_data_dict()

    tfrecord_wrt = tf.python_io.TFRecordWriter('./dataset/train.tfrecord')
    cnt = 0
    for d in dt:
        if 'body' in d.keys():
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


def read_tf_record(filepath='/data/ysc/tensorflow_datasets/scientific_papers/'
                            'arxiv/1.1.1/scientific_papers-test.tfrecord-00001-of-00002'):
    def _parse_record(example_photo):
        features = {
            'inputs': tf.FixedLenFeature((), tf.string),
            'targets': tf.FixedLenFeature((), tf.string)
        }
        parsed_features = tf.parse_single_example(example_photo, features=features)
        return parsed_features

    file_queue = tf.train.string_input_producer([filepath],)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example, features={'inputs': tf.FixedLenFeature([], tf.string),
                                                                    'targets': tf.FixedLenFeature([], tf.string)})
    x, y = features['inputs'], features['targets']
    d = tf.data.TFRecordDataset(filepath)
    dd = d.map(_parse_record)
    return x, y


def xml2json(info_path='../dataset/abstract_info.json', xml_path='../check_data'):

    def ns_tag(*args):
        ns = "/{http://www.tei-c.org/ns/1.0}"

        return './%s%s' % (ns, ns.join(args))

    with open(info_path, 'r') as f:
        abs_info = json.load(f)
    for i, it in enumerate(abs_info):
        xml_file = xml_path + os.sep + str(it['id']) + '.tei.xml'

        dom = etree.parse(xml_file)
        root = dom.getroot()

        body = root.findall(ns_tag('body', 'div'))

        for div in body:
            head = div.find(ns_tag('head'))

            try:
                head_text = etree.tostring(head, encoding='utf-8', method='text')
            except:
                print(i, it['id'], "  div num: ", len(body))
                continue

            ps = div.findall(ns_tag('p'))
            p_text = ""
            if not ps:
                continue
            p_text = " ".join([etree.tostring(p, encoding='utf-8', method='text').decode('utf-8') for p in ps])

            # print(head_text)
            # print(p_text)


        # print("***")

        #break

a = " ". join(['sd'])
print(a)
xml2json()