import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf


model_path = "/home/gitlib/longsumm/output/acl_ss_clean/model.ckpt-"
pre = None
for i in range(0, 70000, 10000):
    ckpt = tf.compat.v1.train.NewCheckpointReader(model_path + str(i))
    key = ckpt.debug_string().decode("utf-8")
    cur = ckpt.get_tensor("pegasus/decoder/layer_14/output/dense/kernel")
    if pre is None:
        pre = cur
    else:
        print('%d--%d: ' % (i-10000, i))
        print(np.abs(cur-pre)/np.abs(pre))
        pre = cur
    print("---"*32)
