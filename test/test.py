import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def test_tf_records(file_path='../dataset/arxiv/'):
    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        name_to_features = {
            "document": tf.io.FixedLenFeature([], tf.string),
            "summary": tf.io.FixedLenFeature([], tf.string),

        }
        example = tf.io.parse_single_example(record, name_to_features)
        return example["document"], example["summary"]

    f = tf.io.gfile.walk(file_path)
    f = next(f)
    print(f)

    files = [file_path + str(it) for it in f[2]]

    ds = tf.data.TFRecordDataset(files)
    print(len(list(ds.as_numpy_iterator())))
    ds = ds.map(_decode_record)
    for i in ds.take(1):
        print(i)



def test_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.test.is_gpu_available()


test_tf_records()
