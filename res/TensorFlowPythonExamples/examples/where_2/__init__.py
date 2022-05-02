import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_b_ = tf.compat.v1.placeholder(dtype=tf.bool, shape=[2], name="Hole")
in_x_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 3], name="Hole")
in_y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 3], name="Hole")
where_ = tf.compat.v1.where(in_b_, in_x_, in_y_)
