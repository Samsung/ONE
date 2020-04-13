import tensorflow as tf

in_ = tf.placeholder(dtype=tf.float32, shape=(1, 8, 8, 3), name="Hole")
op_ = tf.image.resize_bilinear(in_, [16, 16])
