import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.bool, shape=(4, 4), name="Hole")
op_ = tf.compat.v1.logical_not(in_)
