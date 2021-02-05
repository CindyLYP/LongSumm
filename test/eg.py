import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft


def input_fn_builder(data_dir='../bigbird/dataset/bigbird.tfrecords', vocab_model_file="../bigbird/vocab/pegasus.model", max_encoder_length=100,
                     max_decoder_length=20, substitute_newline="<n>", is_training=True,
                     tmp_dir=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example["document"], example["summary"]

  def _tokenize_example(document, summary):
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
      document = tf.strings.regex_replace(document, "\n", substitute_newline)
    # Remove space before special tokens.
    document = tf.strings.regex_replace(document, r" ([<\[]\S+[>\]])", b"\\1")
    document_ids = tokenizer.tokenize(document)
    if isinstance(document_ids, tf.RaggedTensor):
      document_ids = document_ids.to_tensor(0)
    document_ids = document_ids[:max_encoder_length]

    # Remove newline optionally
    if substitute_newline:
      summary = tf.strings.regex_replace(summary, "\n", substitute_newline)
    # Remove space before special tokens.
    summary = tf.strings.regex_replace(summary, r" ([<\[]\S+[>\]])", b"\\1")
    summary_ids = tokenizer.tokenize(summary)
    # Add [EOS] (1) special tokens.
    suffix = tf.constant([1])
    summary_ids = tf.concat([summary_ids, suffix], axis=0)
    if isinstance(summary_ids, tf.RaggedTensor):
      summary_ids = summary_ids.to_tensor(0)
    summary_ids = summary_ids[:max_decoder_length]

    return document_ids, summary_ids

  def input_fn():
    """The actual input function."""
    batch_size = 32

    # Load dataset and handle tfds separately
    split = "train" if is_training else "validation"
    if "tfds://" == data_dir[:7]:
      d = tfds.load('scientific_papers/arxiv', split=split, data_dir='/data/ysc/tensorflow_datasets',
                    shuffle_files=is_training, as_supervised=True)
    else:

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      d = tf.data.TFRecordDataset(data_dir)

      d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    d = d.map(_tokenize_example,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
      d = d.repeat()
    d = d.padded_batch(batch_size, ([max_encoder_length], [max_decoder_length]),
                       drop_remainder=True)
    # For static shape
    for i in d.take(2):
        print(i[0])
        print(i[1])
    return d
  d = input_fn()

input_fn_builder()