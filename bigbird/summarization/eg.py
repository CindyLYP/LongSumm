from bigbird.core import flags
from bigbird.core import modeling
from bigbird.core import utils
from bigbird.summarization import run_summarization
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
from tqdm import tqdm
import sys

FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"): flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)

tf.enable_v2_behavior()

FLAGS.data_dir = "tfds://scientific_papers/pubmed"
FLAGS.attention_type = "block_sparse"
FLAGS.couple_encoder_decoder = True
FLAGS.max_encoder_length = 2048  # on free colab only lower memory GPU like T4 is available
FLAGS.max_decoder_length = 256
FLAGS.block_size = 64
FLAGS.learning_rate = 1e-5
FLAGS.num_train_steps = 10000
FLAGS.attention_probs_dropout_prob = 0.0
FLAGS.hidden_dropout_prob = 0.0
FLAGS.vocab_model_file = "gpt2"


transformer_config = flags.as_dictionary()

from tensorflow.python.ops.variable_scope import EagerVariableStore

container = EagerVariableStore()

with container.as_default():
  model = modeling.TransformerModel(transformer_config)

