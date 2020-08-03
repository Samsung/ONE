import tensorflow as tf

i = tf.constant(0, shape=[1,0], dtype=tf.int32, name='i')
x = tf.compat.v1.placeholder(shape=[1,1], dtype=tf.int32, name='Hole')

c = lambda i: tf.compat.v1.less(tf.compat.v1.size(i[0]), 10)
b = lambda i: tf.concat([i, x], axis=1)

# this loop changs i's shape from [1, 0] -> [1, 1] -> [1, 2] -> ... -> [1, 10]
r = tf.compat.v1.while_loop(c, b, [i], name="While",
                            shape_invariants=[tf.TensorShape([1, None])])

output = tf.compat.v1.identity(r, name="Output")

# by adding the following code, [[1 1 1 1 1 1 1 1 1 1]] and (1, 10) will be printed
#
# with tf.Session() as sess:
#   result = sess.run(r)
#   print(result)
#   print(result.shape)

# with TF 2.3, tf2tflite throws the following error
#
# Exception: venv/tf-2.3/lib/python3.6/site-packages/tensorflow/python/eager/lift_to_graph.py:339:0:
# error: body function result type tensor<1x1xi32> is incompatible with result type tensor<1x0xi32>
# at index 0
# ...
# note: see current operation: %1:2 = "tf.While"(%0, %arg0)
# {body = @_functionalize_body_00, cond = @_functionalize_cond_00, device = "", is_stateless = false, output_shapes = [], parallel_iterations = 10 : i64}
# : (tensor<1x0xi32>, tensor<1x1xi32>) -> (tensor<1x0xi32>, tensor<1x1xi32>)
