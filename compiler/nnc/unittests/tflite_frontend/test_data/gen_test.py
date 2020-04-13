#!/usr/bin/python3
import sys
from distutils.version import LooseVersion, StrictVersion
try:
    import numpy as np
except ImportError:
    print(
        "!! NumPy is not installed, tflite frontend test not generated", file=sys.stderr)
    exit(1)
try:
    import tensorflow as tf
    if (LooseVersion(tf.VERSION) < LooseVersion("1.11.0")):
        raise (Exception("Wrong Version"))
except:
    print(
        "!! Tensorflow v 1.11 not installed, tflite frontend test not generated",
        file=sys.stderr)
    exit(1)

resDir = sys.argv[1]
if resDir[-1] != "/": resDir += "/"

output_shape = [1, 28, 28, 1]
strides = [1, 1, 1, 1]
W = tf.constant(np.ones([7, 7, 1, 1]).astype(np.float32), name="ker_d")

# Create the graph.
X = tf.placeholder(shape=[1, 28, 28, 1], name='input', dtype=tf.float32)
Y = tf.sin(X)

out0 = tf.identity(Y, name="out")
# Filter the input image.
with tf.Session() as sess:
    out = sess.run(
        out0, feed_dict={"input:0": np.ones((1, 28, 28, 1)).astype(np.float32)})
    frozen_graphdef = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ["out"])
    converter = tf.contrib.lite.TocoConverter.from_session(sess, [X], [out0])
    tflite_model = converter.convert()

    open(resDir + "unsupported.tflite", "wb").write(tflite_model)
