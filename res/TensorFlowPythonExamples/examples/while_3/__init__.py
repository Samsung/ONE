import tensorflow as tf

x = tf.compat.v1.placeholder(shape=[1, None], dtype=tf.int32, name='Hole')
i = tf.compat.v1.placeholder(shape=[1, None], dtype=tf.int32, name='Hole_2')


def c(ii):
    rs = tf.compat.v1.shape(ii)
    r1 = rs[1]
    return tf.compat.v1.less(r1, 10)


def b(ii):
    return tf.concat([ii, x], axis=1)


# this loop changes i's shape from [1, 0] -> [1, 1] -> [1, 2] -> ... -> [1, 10]
r = tf.compat.v1.while_loop(
    c, b, [i], name="While", shape_invariants=[tf.TensorShape([1, None])])

output = tf.compat.v1.identity(r, name="Output")

# by adding the following code, [[123 1 2 3 1 2 3 1 2 3]] and (1, 10) will be printed
#
'''
import numpy as np
i_val = np.array([[123]], dtype=np.int32)
x_val = np.array([[1, 2, 3]], dtype=np.int32)
with tf.compat.v1.Session() as sess:
  result = sess.run(r, feed_dict={x:x_val, i:i_val})
  print(result)
  print(result.shape)
'''
