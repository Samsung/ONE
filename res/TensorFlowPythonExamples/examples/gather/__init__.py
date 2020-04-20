import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3, 4), name="Hole")
pos_ = tf.constant([[0, 1, 2], [1, 2, 3]])
op_ = tf.gather(in_, pos_, axis=2)
