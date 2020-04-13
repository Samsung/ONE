import tensorflow as tf

lhs_ = tf.placeholder(dtype=tf.bool, shape=(4, 4), name="Hole")
rhs_ = tf.placeholder(dtype=tf.bool, shape=(4, 4), name="Hole")
op_ = tf.logical_or(lhs_, rhs_)
