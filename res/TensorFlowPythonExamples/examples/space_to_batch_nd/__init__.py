import tensorflow as tf

in_ = tf.compat.v1.placeholder(tf.float32, shape=[1, 2, 2, 1], name="Hole")
bs_ = tf.constant([2, 2], name="Hole")
pd_ = tf.constant([[0, 0], [0, 0]], name="Hole")
op_ = tf.space_to_batch_nd(in_, bs_, pd_)
