import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# this modified example comes from TF API reference
start = 1
limit = 10
delta = 1
range_ = tf.range(start, limit, delta)
