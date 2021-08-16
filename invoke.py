#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import os
import sys

# create model
input_shape = (1, 32, 32, 16)
ksize = [1, 3, 3, 1]
stride = [1, 1, 1, 1]

def infer_tf(x):
  return tf.nn.max_pool_with_argmax(x, ksize, stride, 'VALID', output_dtype=tf.dtypes.int64)

def create_model():
  f = tf.function(infer_tf)
  input_spec = tf.TensorSpec(shape=input_shape, dtype=tf.float32)
  concrete_func = f.get_concrete_function(input_spec)
  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

  converter.allow_custom_ops = True

  tflite_model = converter.convert()
  with open("model.tflite", "wb") as f:
    f.write(tflite_model)
  os.system("./quantize_q16.sh model.tflite")

create_model()

#with open('input.0', 'rb') as binf:
#  input_data = np.frombuffer(binf.read(), dtype=np.int16).reshape(input_shape)
input_data = (np.random.random(input_shape)*10000-5000).astype(dtype = np.float32)

print("input_data", input_data)

ref_output, ref_argmax = infer_tf(input_data)

with open("input.0", 'wb') as binf:
  binf.write(input_data.tobytes('C'))

os.system("./build/compiler/luci-eval-driver/luci_eval_driver model.opt.circle 1 input. output.")

with open('output.1', 'rb') as binf:
  test_argmax = np.frombuffer(binf.read(), dtype=np.int64).reshape(ref_argmax.shape)

print("reference indeces: ", ref_argmax)
print("test indeces: ", test_argmax)

for i in range(ref_argmax.shape[1]):
  for j in range(ref_argmax.shape[2]):
    for k in range(ref_argmax.shape[3]):
      if ref_argmax[0, i, j, k] != test_argmax[0, i, j, k]:
        print("diff for idx ", (i, j,k), "ref:", ref_argmax[0, i, j, k], "test", test_argmax[0, i, j, k])

