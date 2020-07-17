import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(6), name="Hole")
op_ = tf.compat.v1.unique(in_)
