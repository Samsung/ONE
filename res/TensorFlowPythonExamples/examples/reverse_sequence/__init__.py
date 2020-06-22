import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 8), name="Hole")
op_ = tf.compat.v1.reverse_sequence(in_, [7, 2, 3, 5], seq_axis=1, batch_axis=0)
