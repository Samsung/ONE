#!/usr/bin/python3
import onnx
import onnxruntime
import numpy as np
import sys

if len(sys.argv) < 3:
  print("usage: ./compare.py <reference model> <test model>")
  exit(1)

ref_session=onnxruntime.InferenceSession(sys.argv[1])

test_session=onnxruntime.InferenceSession(sys.argv[2])

input_data = {}
num_inputs = len(ref_session.get_inputs())
assert(num_inputs == len(test_session.get_inputs()))
generator = np.random.default_rng()
for ref_spec, test_spec in zip(ref_session.get_inputs(), test_session.get_inputs()):
  assert(ref_spec.name == test_spec.name)
  assert(ref_spec.type == test_spec.type)
  assert(ref_spec.shape == test_spec.shape)
  input_data[ref_spec.name] = generator.random(ref_spec.shape, dtype=np.float32)

ref_output_data = ref_session.run(None, input_data)
test_output_data = test_session.run(None, input_data)

print("ref: ", ref_output_data)
print("test: ", test_output_data)

for ref_output, test_output in zip(ref_output_data, test_output_data):
  assert(ref_output.shape == test_output.shape)
  print("diff: ", ref_output - test_output)
  norm_diff = np.max(abs((ref_output - test_output)/ref_output))
  print("norm diff: ", np.max(abs((ref_output - test_output)/ref_output)))
  if norm_diff > 1e-4:
    exit(1)

