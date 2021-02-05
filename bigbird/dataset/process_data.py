import json
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def write_record():
    # Read image raw data, which will be embedded in the record file later.
    with open('bigbird_train.json', 'r') as f:
        ds = json.load(f)

    writer = tf.io.TFRecordWriter('bigbird.tfrecords')
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

# write_record()



def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),

    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example["document"], example["summary"]

ds = tf.data.TFRecordDataset('bigbird.tfrecords')
ds = ds.map(_decode_record)
for i in ds.take(3):
    print(repr(i))

dataset = ds.repeat(5).shuffle(1024).batch(32)

dataset = dataset.repeat(2)
dataset = dataset.batch(4) # Batch size to use

iterator = dataset.make_one_shot_iterator()
batch_features, batch_labels = iterator.get_next()
