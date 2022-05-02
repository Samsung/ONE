import tensorflow as tf

tf.compat.v1.disable_eager_execution()

lhs_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
rhs_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.compat.v1.minimum(lhs_, rhs_)
