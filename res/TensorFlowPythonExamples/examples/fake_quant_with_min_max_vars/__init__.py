import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 32, 32, 3), name="Hole")

filters = np.random.uniform(low=-1., high=1, size=[5, 5, 3, 32]).astype(np.float32)
strides = (1, 2, 2, 1)
cv_ = tf.compat.v1.nn.conv2d(in_, filters, strides, "VALID", data_format="NHWC")

op_ = tf.compat.v1.fake_quant_with_min_max_vars(cv_, 0.0, 1.0, 8, False)
'''
NOTE:
'fake_quant_with_min_max_vars' is converted to QUANTIZE-DEQUANTIZE in tflite.
To produce tflite with FAKE_QUANT Op, you need to change tf2tfliteV2.py with

converter.experimental_new_converter = False

and then run

python3 ../../compiler/tf2tfliteV2/tf2tfliteV2.py --v2 --graph_def \
-i ./fake_quant_with_min_max_vars.pbtxt \
-o ./fake_quant_with_min_max_vars.tflite \
-I Hole \
-O FakeQuantWithMinMaxVars
'''
