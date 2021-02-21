import json
import os
# from xml.dom.minidom import parse
from xml.etree.ElementTree import parse
from lxml import etree
from xml.etree.ElementTree import iterparse
import pandas as pd
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
from tqdm import tqdm
import numpy as np

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


def xml2json(info_path='../dataset/json_data/pdf_info.json', xml_path='../dataset/xml'):
    def ns_tag(*args):
        ns = "/{http://www.tei-c.org/ns/1.0}"

        return './%s%s' % (ns, ns.join(args))

    with open(info_path, 'r') as f:
        abs_info = json.load(f)
    remain = []
    cnt = 0
    for i, it in enumerate(abs_info):
        xml_file = xml_path + os.sep + str(it['id']) + '.tei.xml'

        if not os.path.exists(xml_file):
            cnt += 1
            continue
        dom = etree.parse(xml_file)
        root = dom.getroot()

        try:
            abstract = str(root.find(ns_tag('abstract', 'p')).xpath('text()')[0])
        except:
            print("no abstract found in ", xml_file)
            exit(1)
        abs_info[i]['abstract'] = abstract

        body = root.findall(ns_tag('body', 'div'))
        section_name, section_content = [], []

        for div in body:
            head = div.find(ns_tag('head'))
            ps = div.findall(ns_tag('p'))
            if not ps:
                print('no content found in ', i, it['id'])
                continue
            try:
                section_name.append(etree.tostring(head, encoding='utf-8', method='text').decode('utf-8').lower())
            except:
                print("no head found in ",xml_file)
                exit(1)
            section_content.append(" ".join([etree.tostring(p, encoding='utf-8', method='text').
                                            decode('utf-8') for p in ps]))
        abs_info[i]['section_name'] = section_name
        abs_info[i]['section_content'] = section_content
        remain.append(abs_info[i])
    with open("../dataset/xml/remain.json", 'w') as f:
        json.dump(remain, f)
    print(cnt)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_record(features: dict, feat_type: dict, file_path):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8', 'ignore')]))

    feat_name = list(features.keys())
    n_example = len(features[feat_name[0]])
    writer = tf.io.TFRecordWriter(file_path)
    for it in range(n_example):
        feature = {}
        for name in feat_name:
            if feat_type[name] == 'string':
                tf_feat = _bytes_feature(features[name][it])
            feature[name] = tf_feat
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(tf_example.SerializeToString())

    writer.close()


def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),

    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example["document"], example["summary"]


def read_tf_record(file_path):  # "need '/' at the end of the path eg: /home/dataset/ "
    f = tf.io.gfile.walk(file_path)
    f = next(f)
    print("find files: ", f)
    files = [file_path + str(it) for it in f[2]]

    ds = tf.data.TFRecordDataset(files)
    print("total num: ", len(list(ds.as_numpy_iterator())))
    ds = ds.map(_decode_record)

    # try this to print an example
    for i in ds.take(10):
        print("Feature:\n{}\n\nLabel:\n{}\n\n".format(i[0].numpy().decode('utf-8'), i[1].numpy().decode('utf-8')))

    return ds


def gen_long_arxiv_data(file_path='../bigbird/dataset/arxiv/train.tfrecord-'):
    d = tfds.load('scientific_papers/arxiv', split="train", data_dir='/home/tensorflow_datasets',
                  as_supervised=True)
    threhold = [2000, 3000, 4000, 5000]
    d = d.as_numpy_iterator()
    feat_type = {'document': 'string',
                 'summary': 'string'}
    articles = []
    summaries = []
    print("===== write into the tf records =====")
    cnt = 0
    for it in tqdm(d):
        article = it[0].decode('utf-8')
        summary = it[1].decode('utf-8')
        l = len(summary.split())
        for j in range(4):
            if threhold[j] > 0 and j * 100 < l <= (j + 1) * 100:
                articles.append(article), summaries.append(summary)
                threhold[j] -= 1
                flag = False
                break
        if l > 400:
            articles.append(article), summaries.append(summary)
        if len(articles) >= 4096:
            print("write into record-%d" % cnt)
            features = {'document': articles,
                        'summary': summaries}
            write_record(features, feat_type, file_path + str(cnt))
            print("write finish")
            cnt += 1
            articles.clear(), summaries.clear()

    if articles:
        features = {'document': articles,
                    'summary': summaries}
        write_record(features, feat_type, file_path + str(cnt))

    print("=====           finish           =====")


file_path = '../dataset/gen_data/train.tfrecord-'


def ex_data():
    d = []
    reh = 50
    with open("../dataset/json_data/gen_data.json", 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            d.append(json.loads(line))
            line = f.readline()
    val_acl = d[:reh]
    val_arxiv = d[len(d)-1000:]
    data = d[reh:len(d)-1000]
    le = len(data)

    feat_type = {'document': 'string',
                 'summary': 'string'}

    def transfer_feat(rd):
        doc = [it[0] for it in rd]
        summ = [it[1] for it in rd]

        return {'document': doc, 'summary': summ}

    feat = transfer_feat(val_acl)
    write_record(feat, feat_type, "../dataset/gen_data/val_acl.tfrecord")
    print("write val acl")
    feat = transfer_feat(val_arxiv)
    write_record(feat, feat_type, "../dataset/gen_data/val_arxiv.tfrecord")
    print("write val arxiv")
    cnt = 0
    for i in tqdm(range(0, le, 6000)):
        feat = transfer_feat(data[i:min(i+3000, le)])
        write_record(feat, feat_type, "../dataset/gen_data/train.tfrecord-%d"%cnt)
        cnt += 1
    print(cnt)


def gen_acl_ss_data():

    with open("../dataset/json_data/acl515.json", 'r') as f:
        d = json.load(f)
    tot_data = []
    cnt = 0
    for it in d:
        if "shortscience" in it['source_website']:
            continue
        doc = it['abstract'] + " ".join(it['section_content'])
        summ = " ".join(it['summary'])

        if len(summ.split()) <= 100:
            cnt += 1
            continue

        tot_data.append({"document": doc,
                         "summary": summ})
    print("acl drop num: ", cnt)

    union_data = []
    with open('../dataset/json_data/union_add.json', 'r') as f:
        line = f.readline()
        while line:
            union_data.append(json.loads(line))
            line = f.readline()
    cnt = 0

    def drop_url(s):
        return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-|\#)*\b', '[url]', s, flags=re.MULTILINE)
    for it in union_data:
        doc = it['page']['text']
        summ = drop_url(it['summary'])
        if len(doc.split()) < 1000 or len(summ.split()) < 50:
            cnt += 1
            continue
        tot_data.append({"document": doc,
                         "summary": summ})

    print("short science drop num: ", cnt)
    print("=="*32)
    print("total data: ", len(tot_data))
    with open("../dataset/json_data/acl_ss.json", 'w') as f:
        json.dump(tot_data, f)
    print("finish")


# gen_acl_ss_data()