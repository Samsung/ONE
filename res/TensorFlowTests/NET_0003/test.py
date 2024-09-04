# TF_SMALL_NET_0003/test.pbtxt is create with below script

# Version info
#  - Tensorflow : 1.13.1
#  - Python : 3.5.2

import tensorflow as tf

input0 = tf.placeholder(tf.float32, [1, 3, 3, 5])
filter0 = tf.constant(1.0, shape=[2, 2, 5, 1])
conv = tf.nn.conv2d(input0, filter=filter0, strides=[1, 1, 1, 1], padding='SAME')
fbn = tf.nn.fused_batch_norm(conv,
                             scale=[1.0],
                             offset=[0.0],
                             mean=[0.0],
                             variance=[1.0],
                             is_training=False)

print(tf.get_default_graph().as_graph_def())
