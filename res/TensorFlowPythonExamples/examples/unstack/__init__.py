import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[4, 2, 3, 4], name="Hole")
unpack_ = tf.compat.v1.unstack(in_, axis=0)
