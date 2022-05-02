import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 8, 8, 3), name="Hole")
op_ = tf.compat.v1.image.resize_nearest_neighbor(in_, [16, 16])
