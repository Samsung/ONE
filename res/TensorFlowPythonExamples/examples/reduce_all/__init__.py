import tensorflow as tf

tf.compat.v1.disable_eager_execution()

input_ = tf.compat.v1.placeholder(dtype=tf.bool, shape=(2, 4), name="Hole")
op_ = tf.compat.v1.reduce_all(input_, axis=1, keepdims=False)
