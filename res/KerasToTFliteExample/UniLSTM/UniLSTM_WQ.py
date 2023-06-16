import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(60, 128)),
    tf.keras.layers.LSTM(64,return_sequences=True), # return_sequences=False is default, which produces LSTM->StridedSlice tflite model
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.summary()

run_model = tf.function(lambda x: model(x))

# This is important, let's fix the input size.
BATCH_SIZE = 1
X = 60
Y = 128
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, X, Y], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT] # For Weight Quantization(a.k.a Hybrid Quantization)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converted_model = converter.convert()
save_to = "unilstm_wq.tflite"
if save_to is not None:
    with open(save_to, 'wb') as tf_lite_file:
        tf_lite_file.write(converted_model)
