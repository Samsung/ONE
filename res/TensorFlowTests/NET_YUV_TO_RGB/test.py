#used to create test.pbtxt
# Version info
#  - Tensorflow : 2.2.0
#  - Python : 3.7.7

import tensorflow as tf
# Construct a basic model.
root = tf.train.Checkpoint()
root.f = tf.function(lambda x: tf.image.yuv_to_rgb(x))
# Create the concrete function.
input_data = tf.constant(1., shape=[1, 4, 4, 3])
# dim = tf.constant(12,shape=[2])
concrete_func = root.f.get_concrete_function(input_data)
tf.io.write_graph(concrete_func.graph.as_graph_def(), '.', 'test.pbtxt')
