# TF_SMALL_NET_0004/test.pbtxt is create with below script

# Version info
#  - Tensorflow : 1.13.1
#  - Python : 3.5.2

import tensorflow as tf

input0 = tf.placeholder(tf.float32, [1, 3, 3, 5])
filter0 = tf.constant(1.0, shape=[2, 2, 5, 2])
dconv = tf.nn.depthwise_conv2d(input0, filter0, [1, 1, 1, 1], 'SAME')
const = tf.constant(2.0, shape=[10])
fbn = tf.nn.fused_batch_norm(x=dconv,
                             scale=const,
                             offset=const,
                             mean=const,
                             variance=const,
                             is_training=False)

print(tf.get_default_graph().as_graph_def())
