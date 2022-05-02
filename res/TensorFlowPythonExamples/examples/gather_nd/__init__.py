import tensorflow as tf

tf.compat.v1.disable_eager_execution()

param_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 2, 2), name="Hole")
indices_ = tf.constant([[0, 1], [1, 0]])
op_ = tf.gather_nd(param_, indices_)
