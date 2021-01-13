import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 16, 160, 160), name="Hole")

upper_ = tf.compat.v1.constant(6.)
lower_ = tf.compat.v1.constant(0.)

min_ = tf.compat.v1.minimum(in_, upper_)
max_ = tf.compat.v1.maximum(min_, lower_)
'''
python ../../compiler/tf2tfliteV2/tf2tfliteV2.py --v1 \
-i minimum-maximum.pbtxt \
-o minimum-maximum.tflite \
-I Hole -O Maximum
'''
