import json
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def write_record():
    # Read image raw data, which will be embedded in the record file later.
    with open('bigbird_train.json', 'r') as f:
        ds = json.load(f)

    writer = tf.python_io.TFRecordWriter('bigbird.tfrecords')
    for i in range(len(ds['summary'])):
        doc = ds['document'][i]
        summ = ds['summary'][i]
        feature = {
        'document': _bytes_feature(doc),
        'summary': _bytes_feature(summ)}

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("finish")
write_record()