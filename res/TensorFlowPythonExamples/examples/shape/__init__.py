import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="Hole")
op_ = tf.compat.v1.shape(in_)
