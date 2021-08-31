import tensorflow as tf

input_path = "./inception_v3.pb"
output_path = "./inception_v3.tflite"

input_shapes = {"input": [0, 299, 299, 3]}
input_nodes = ["input"]
output_nodes = ["InceptionV3/Predictions/Reshape_1"]

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    input_path, input_nodes, output_nodes, input_shapes)
converter.allow_custom_ops = True
tflite_model = converter.convert()
open(output_path, "wb").write(tflite_model)
