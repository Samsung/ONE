import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.compat.v1.matrix_band_part(in_, 1, -1)
