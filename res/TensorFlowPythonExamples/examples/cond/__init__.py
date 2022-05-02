import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="HoleX")
y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="HoleY")
z_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="HoleZ")


def fn1(a, b):
    return tf.math.multiply(a, b, name="HoleM")


def fn2(a, b):
    return tf.math.add(a, b, name="HoleA")


pr_ = tf.compat.v1.placeholder(tf.bool, shape=[], name="HoleC")
op_ = tf.cond(pr_, lambda: fn1(x_, y_), lambda: fn2(y_, z_), name="Cond")
re_ = tf.identity(op_, name="HoleR")
