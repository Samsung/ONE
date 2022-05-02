import tensorflow as tf

tf.compat.v1.disable_eager_execution()

i = tf.compat.v1.constant(0, name="Hole")

c = lambda i: tf.compat.v1.less(i, 10)
b = lambda i: tf.compat.v1.add(i, 1)
r = tf.compat.v1.while_loop(c, b, [i], name="While")
