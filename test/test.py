import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
import pandas as pd
from tqdm import tqdm
import numpy as np



def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),

    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example["document"], example["summary"]



file_path='../bigbird/dataset/arxiv/train.tfrecord-'
files = [file_path + str(it) for it in range(8)]

ds = tf.data.TFRecordDataset(files)
print(len(list(ds.as_numpy_iterator())))
ds = ds.map(_decode_record)

