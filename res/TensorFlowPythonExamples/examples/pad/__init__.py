import tensorflow as tf

tensor_ = tf.placeholder(dtype=tf.float32, shape=(2, 3), name="Hole")
paddings_ = tf.constant([[1, 1], [2, 2]], name="Hole")
op_ = tf.pad(tensor_, paddings_)
