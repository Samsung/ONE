import tensorflow as tf

x_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="HoleX")
y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="HoleY")
z_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="HoleZ")


def fn01(a, b):
    return tf.math.multiply(a, b, name="Hole0M")


def fn02(a, b):
    return tf.math.add(a, b, name="Hole0A")


def fn1(c, x, y, z):
    return tf.cond(c, lambda: fn01(x, y), lambda: fn02(y, z), name="Cond0")


def fn2(a, b):
    return tf.math.add(a, b, name="HoleA")


pr_ = tf.compat.v1.placeholder(tf.bool, shape=[], name="HoleC")
op_ = tf.cond(pr_, lambda: fn1(pr_, x_, y_, z_), lambda: fn2(y_, z_), name="Cond")
re_ = tf.identity(op_, name="HoleR")
