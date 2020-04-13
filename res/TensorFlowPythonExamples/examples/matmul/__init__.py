import tensorflow as tf

lhs_ = tf.placeholder(dtype=tf.float32, shape=(3, 4), name="Hole")
rhs_ = tf.constant(dtype=tf.float32, shape=(4, 4), name="Hole", value=1.0)
op_ = tf.matmul(lhs_, rhs_)
