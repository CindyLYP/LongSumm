import json
import os
import tensorflow as tf


def json2tfrecord(path='../train_json'):
    _, _, files = next(os.walk(path))
    for file in files:
        with open(path+'/'+file, 'r') as fp:
            raw_d = json.load(fp)
            print(raw_d)


            break


json2tfrecord()
