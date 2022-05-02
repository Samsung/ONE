import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(tf.float32, shape=[1, 2, 2, 1], name="Hole")
pd_ = tf.constant([[0, 0], [0, 0]], name="Hole")
op_ = tf.space_to_batch(in_, pd_, 2)
