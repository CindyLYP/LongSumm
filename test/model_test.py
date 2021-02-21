import tensorflow as tf


model_path = "/home/gitlib/longsumm/output/acl_ss_part/model.ckpt-"

for i in range(0, 3600, 600):
    ckpt = tf.compat.v1.train.NewCheckpointReader(model_path + str(i))
    key = ckpt.debug_string().decode("utf-8")
    print(ckpt.get_tensor("pegasus/decoder/layer_14/output/dense/kernel"))
    print("---"*32)
