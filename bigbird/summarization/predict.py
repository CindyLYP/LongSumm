import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
print(os.environ["CUDA_VISIBLE_DEVICES"])

from bigbird.core import flags
from bigbird.core import modeling
from bigbird.summarization import run_summarization
import tensorflow.compat.v2 as tf
from tensorflow.python.ops.variable_scope import EagerVariableStore
import tensorflow_text as tft
from tqdm import tqdm
import sys
from model.session_rank import slide_window
import json
from scripts.eval import rouge_metric


FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"):
    flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)
# tf.enable_v2_behavior()


FLAGS.max_encoder_length = 2048
FLAGS.max_decoder_length = 256
FLAGS.vocab_model_file = "/home/gitlib/longsumm/bigbird/vocab/pegasus.model"
FLAGS.eval_batch_size = 4
FLAGS.substitute_newline = "<n>"
ckpt_path = '/home/gitlib/longsumm/output/acl_ss_part/model.ckpt-50000'
pred_out = '/home/gitlib/longsumm/output/acl_ss_part/pred.txt'
pred_in = '/home/gitlib/longsumm/dataset/acl_ss_clean/pred/pred.json'

tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(FLAGS.vocab_model_file, "rb").read())

num_pred_steps = 5


def input_fn(document):

    def _tokenize_example(doc):
        if FLAGS.substitute_newline:
            doc = tf.strings.regex_replace(doc, "\n", FLAGS.substitute_newline)
        doc = tf.strings.regex_replace(doc, r" ([<\[]\S+[>\]])", b"\\1")
        document_ids = tokenizer.tokenize(doc)
        if isinstance(document_ids, tf.RaggedTensor):
            dim = document_ids.shape[0]
            document_ids = document_ids.to_tensor(0, shape=(dim, FLAGS.max_encoder_length))
        else:
            document_ids = document_ids[:, :FLAGS.max_encoder_length]

        return document_ids

    feats = slide_window(document)
    d = _tokenize_example(feats)

    return d


def main():
    transformer_config = flags.as_dictionary()
    container = EagerVariableStore()
    with container.as_default():
        model = modeling.TransformerModel(transformer_config)

    with open(pred_in, 'r') as f:
        dataset = json.load(f)

    @tf.function(experimental_compile=True)
    def fwd_only(features):
        (llh, logits, pred_ids), _ = model(features, target_ids=None,
                                           training=False)
        return llh, logits, pred_ids

    ex = input_fn(dataset[10]['document'])
    with container.as_default():
        llh, logits, pred_ids = fwd_only(ex)

    print('==== build model')

    ckpt_reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
    loaded_weights = []

    for v in tqdm(model.trainable_weights, position=0):
        try:
            val = ckpt_reader.get_tensor(v.name[:-2])
        except:
            val = v.numpy()
        loaded_weights.append(val)
    model.set_weights(loaded_weights)
    print("==== load model weights")

    cnt = 0
    s1, s2 = [], []

    with open(pred_out, 'w', encoding='utf-8') as f:

        for i, ex in enumerate(dataset):
            print("### Example %d" % i)
            document, summary = ex['document'], ex['summary']
            doc_ids = input_fn(document)
            _, _, pred_ids = fwd_only(doc_ids)
            pred_sents = tokenizer.detokenize(pred_ids)
            pred_sents = tf.strings.regex_replace(pred_sents, r"([<\[]\S+[>\]])", b" \\1")
            if transformer_config["substitute_newline"]:
                pred_sents = tf.strings.regex_replace(pred_sents, transformer_config["substitute_newline"], "\n")

            pred_list = [s.numpy().decode('utf-8') for s in pred_sents]
            pred_summary = " ".join([s.numpy().decode('utf-8') for s in pred_sents])
            s1.append(pred_summary), s2.append(summary)

            output_info = "Article:\n %s\n\n Ground truth summary:\n %s\n\n Predicted summary:\n %s\n\n" % \
                          (document, summary, pred_summary)
            single_rouge = rouge_metric([pred_summary], [summary])
            for key in sorted(single_rouge.keys()):
                output_info += "%s = %.4f " % (key, single_rouge[key])

            output_info += "\n" + "==="*32 + "\n"
            f.write(output_info)
            print(output_info)
            for it in pred_list:
                print(it)
                print("**"*32)
        res = rouge_metric(s1, s2)
        avg_rouge = "average rouge score: \n"
        for key in sorted(single_rouge.keys()):
            avg_rouge += "%s = %.4f\n" % (key, single_rouge[key])
        f.write(avg_rouge)
        print(avg_rouge)


if __name__=="__main__":
    main()