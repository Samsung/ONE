import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(3, 3), name="Hole")

op_ = tf.compat.v1.layers.flatten(in_)
