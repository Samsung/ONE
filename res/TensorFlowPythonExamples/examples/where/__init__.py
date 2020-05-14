import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.bool, shape=[2], name="Hole")
where_ = tf.compat.v1.where(in_)
