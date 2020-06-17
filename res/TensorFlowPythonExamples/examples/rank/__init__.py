import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4, 3, 3), name="Hole")
rank_ = tf.compat.v1.rank(in_)
