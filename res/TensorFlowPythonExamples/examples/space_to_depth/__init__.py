import tensorflow as tf

in_ = tf.compat.v1.placeholder(tf.float32, shape=[1, 2, 2, 1], name="Hole")
op_ = tf.nn.space_to_depth(in_, 2)
