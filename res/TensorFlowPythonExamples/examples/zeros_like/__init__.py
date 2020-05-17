import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.zeros_like(in_)
