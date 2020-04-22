import tensorflow as tf

x_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 2, 3), name="HoleX")
y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 2, 3), name="HoleY")


def fn1(a):
    return tf.compat.v1.unstack(a, axis=0, name="HoleU")


pr_ = tf.compat.v1.placeholder(tf.bool, shape=[], name="HoleC")
op_ = tf.cond(pr_, lambda: fn1(x_), lambda: fn1(y_), name="Cond")
re_ = tf.identity(op_, name="HoleR")
