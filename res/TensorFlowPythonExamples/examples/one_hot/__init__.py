import tensorflow as tf

indice_ = tf.compat.v1.placeholder(tf.int32, shape=(1, 2, 3, 4), name='Hole')
depth_ = tf.compat.v1.placeholder(tf.int32, shape=(), name='Hole')
on_value_ = tf.compat.v1.placeholder(tf.int32, shape=(), name='Hole')
off_value_ = tf.compat.v1.placeholder(tf.int32, shape=(), name='Hole')
op_ = tf.one_hot(indices=indice_,
                 depth=depth_,
                 on_value=on_value_,
                 off_value=off_value_,
                 axis=-1)
