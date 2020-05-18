import tensorflow as tf

start = 3
limit = 18
delta = 3
range_ = tf.range(start, limit, delta)  # [3, 6, 9, 12, 15]
