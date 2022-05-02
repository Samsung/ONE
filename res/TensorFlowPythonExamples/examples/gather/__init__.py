import tensorflow as tf

tf.compat.v1.disable_eager_execution()

param_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 2, 3, 4), name="Hole")
indices_ = tf.constant([1, 2])
op_ = tf.gather(param_, indices_, axis=2)
