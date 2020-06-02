import tensorflow as tf

arg = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.math.l2_normalize(arg)
