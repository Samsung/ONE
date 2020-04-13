import tensorflow as tf

lhs_ = tf.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
rhs_ = tf.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.greater_equal(lhs_, rhs_)
