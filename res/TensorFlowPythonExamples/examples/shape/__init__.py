import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="Hole")
op_ = tf.compat.v1.shape(in_)
