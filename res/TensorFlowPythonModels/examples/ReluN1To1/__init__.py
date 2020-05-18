import tensorflow as tf

in1_ = tf.compat.v1.placeholder(tf.float32, shape=(2, 3, 4), name="Hole")
in2_ = tf.constant(-1.0, dtype=tf.float32, shape=[], name="Hole")
in3_ = tf.math.maximum(x=in1_, y=in2_, name="Hole")
in4_ = tf.constant(1.0, dtype=tf.float32, shape=[], name="Hole")
out_ = tf.math.minimum(x=in3_, y=in4_, name="Relu1")

# tf2tfliteV2 will convert this network to RELU_N1_TO_1 Op
# example)
'''
python ../../compiler/tf2tfliteV2/tf2tfliteV2.py \
--v1 \
--input_path ./ReluN1To1.pbtxt \
--output_path ./ReluN1To1.tflite \
--input_arrays Hole \
--output_arrays Relu1
'''
