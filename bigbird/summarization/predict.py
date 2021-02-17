from bigbird.core import flags
from bigbird.core import modeling
from bigbird.core import utils
from bigbird.summarization import run_summarization
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops.variable_scope import EagerVariableStore
import tensorflow_text as tft
from tqdm import tqdm
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"):
    flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)
# tf.enable_v2_behavior()


FLAGS.data_dir = "/home/gitlib/longsumm/dataset/extract_arxiv/"
FLAGS.max_encoder_length = 2048  # on free colab only lower memory GPU like T4 is available
FLAGS.max_decoder_length = 608
FLAGS.vocab_model_file = "pegasus"
FLAGS.eval_batch_size = 4
ckpt_path = '/home/gitlib/longsumm/output/extract_arxiv/model.ckpt-30000'
pred_dir = '/home/gitlib/longsumm/output/extract_arxiv/pred.txt'

num_pred_steps = 5


def main():
    transformer_config = flags.as_dictionary()
    container = EagerVariableStore()
    with container.as_default():
        model = modeling.TransformerModel(transformer_config)

    input_fn = run_summarization.input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        max_decoder_length=FLAGS.max_decoder_length,
        substitute_newline=FLAGS.substitute_newline,
        is_training=False,
        batch_size=FLAGS.eval_batch_size)
    dataset = input_fn()
    for i in dataset.take(1):
        ex = i

    @tf.function(experimental_compile=True)
    def fwd_bwd(features, labels):
        with tf.GradientTape() as g:
            (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                               training=True)
            loss = run_summarization.padded_cross_entropy_loss(
                logits, labels,
                transformer_config["label_smoothing"],
                transformer_config["vocab_size"])
        grads = g.gradient(loss, model.trainable_weights)
        return loss, llh, logits, pred_ids, grads

    @tf.function(experimental_compile=True)
    def fwd_only(features, labels):
        (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                           training=False)
        return llh, logits, pred_ids

    with container.as_default():
        llh, logits, pred_ids = fwd_only(ex[0], ex[1])
    print('build model finish')

    ckpt_reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
    loaded_weights = []

    for v in tqdm(model.trainable_weights, position=0):
        try:
            val = ckpt_reader.get_tensor(v.name[:-2])
        except:
            val = v.numpy()
        loaded_weights.append(val)

    model.set_weights(loaded_weights)

    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(FLAGS.vocab_model_file, "rb").read())

    cnt = 0

    with open(pred_dir, 'w', encoding='utf-8') as f:
        for batch_example in dataset.take(num_pred_steps):
            _, _, pred_ids = fwd_only(batch_example[0], batch_example[1])
            examples = [tokenizer.detokenize(batch_example[0]),
                        tokenizer.detokenize(batch_example[1]),
                        tokenizer.detokenize(pred_ids)]
            examples = [tf.strings.regex_replace(exam, r"([<\[]\S+[>\]])", b" \\1") for exam in examples]

            if transformer_config["substitute_newline"]:
                examples = [tf.strings.regex_replace(exam, transformer_config["substitute_newline"], "\n")
                            for exam in examples]
            for a, b, c in zip(examples[0], examples[1], examples[2]):
                print("Example %d" % cnt)
                cnt += 1
                text = 'Article:\n %s\n\n Ground truth summary:\n %s\n\n Predicted summary:\n %s\n\n' % (
                    a.numpy().decode('utf-8'), b.numpy().decode('utf-8'), c.numpy().decode('utf-8'))
                f.write(text)
                print(text)
                print("==" * 32)


if __name__=="__main__":
    main()