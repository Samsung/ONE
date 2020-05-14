import tensorflow as tf

lhs_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
rhs_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.compat.v1.floormod(lhs_, rhs_)
