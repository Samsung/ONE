import tensorflow as tf

in_b_ = tf.compat.v1.placeholder(dtype=tf.bool, shape=[3], name="Hole")
in_x_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 1], name="Hole")
in_y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 3], name="Hole")
where_v2_ = tf.compat.v1.where_v2(in_b_, in_x_, in_y_)
