import tensorflow as tf

#i = tf.compat.v1.constant(0)
#x = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2, 3), name="Hole")

#c = lambda i, x: tf.less(i, 10)
#b = lambda i, x: (tf.add(i, 1), tf.add(x, 10))
#r = tf.while_loop(c, b, [i, x], name="While")
#re_ = tf.identity(r[1], name="HoleR")

i = tf.compat.v1.constant(0, name="Hole")

c = lambda i: tf.equal(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i], name="While")
re_ = tf.identity(r, name="HoleR")

sess = tf.compat.v1.Session()
tf.io.write_graph(sess.graph, 'tmp/my-model', 'train.pbtxt')
