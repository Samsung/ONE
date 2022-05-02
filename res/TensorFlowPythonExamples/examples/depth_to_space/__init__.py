import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, 1, 4], name="Hole")
op_ = tf.nn.depth_to_space(in_, 2)
