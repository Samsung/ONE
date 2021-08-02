import tensorflow as tf

i = tf.compat.v1.constant(0, name='Hole')


def c(a):
    return tf.compat.v1.less(a, 10)


def b(a):
    ci = lambda inp: tf.compat.v1.less(inp, 10)
    bi = lambda inp: tf.compat.v1.add(inp, 1)
    w = tf.compat.v1.while_loop(ci, bi, [a], name="WhileI")
    return tf.identity(w)


r = tf.compat.v1.while_loop(c, b, [i], name="While")
output = tf.identity(r, name="output")
'''
python3 ../../compiler/tf2tfliteV2/tf2tfliteV2.py --v2 --graph_def \
-i ./while_4.pbtxt \
-o ./while_4.tflite \
-I Hole \
-O output
'''
