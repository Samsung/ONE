import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 6), name="unique_in")
op_ = tf.compat.v1.math.argmin(in_)
