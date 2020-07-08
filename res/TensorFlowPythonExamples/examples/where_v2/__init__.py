import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.bool, shape=[2], name="Hole")
where_v2_ = tf.compat.v1.where_v2(in_)
