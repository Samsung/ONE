import tensorflow as tf

in1_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 8, 8, 1), name="Hole")
in2_ = tf.compat.v1.square(in1_, name="Hole1")
in3_ = tf.compat.v1.nn.avg_pool(in2_, (2, 2), (1, 1), "VALID", name="Hole2")
out_ = tf.compat.v1.sqrt(in3_, name="L2Pool2D")

# this example with tf2tfliteV2 will convert .pbtxt model to .tflite with one L2_POOL_2D
'''
python ../../compiler/tf2tfliteV2/tf2tfliteV2.py \
--v1 \
--input_path ./L2Pool2D.pbtxt \
--output_path ./L2Pool2D.tflite \
--input_arrays Hole \
--output_arrays L2Pool2D
'''
