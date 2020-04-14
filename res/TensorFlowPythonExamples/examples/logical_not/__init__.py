import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.bool, shape=(4, 4), name="Hole")
op_ = tf.compat.v1.logical_not(in_)
