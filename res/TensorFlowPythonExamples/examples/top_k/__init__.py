import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[4], name="Hole")
op_ = tf.compat.v1.math.top_k(in_, k=1)
