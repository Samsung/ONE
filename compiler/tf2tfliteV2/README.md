# tf2tfliteV2

_tf2tfliteV2_ is a TensorFlow to TensorFlow Lite model Converter.

## Where does V2 come from?
Even though we already have _tf2tflite_, we cannot cover all operators in TensorFlow. To expand coverage, we introduce _tf2tfliteV2_ which internally uses `TensorFlow Lite Converter` (by Google).

## Prerequisite
- Frozen graph from TensorFlow 1.13.1 in binary (`*.pb`) or text (`*.pbtxt`) format
- Desired version of TensorFlow (You can use python virtualenv, docker, etc.)

## Example
```
python tf2tfliteV2.py \
> --v1 \
> -i frozen_graph.pb -o converted.tflite
> -I model_inputs -O model_outputs
```
```
python tf2tfliteV2.py \
> --v1 \
> --input_path=frozen_graph.pb \
> --output_path=converted.tflite \
> --input_arrays=model_inputs \
> --output_arrays=model_outputs

```
```
python tf2tfliteV2.py \
> --v2 \
> --input_path=frozen_graph.pbtxt \
> --output_path=converted.tflite \
> --input_arrays=model_inputs \
> --output_arrays=model_outputs
```
```
python tf2tfliteV2.py \
> --v2 \
> --input_path=multiple_output_graph.pb \
> --output_path=converted.tflite \
> --input_arrays=model_inputs \
> --output_arrays=output,output:1,output:2
```

## Optional arguments
```
  -h, --help            show this help message and exit
  --v1                  Use TensorFlow Lite Converter 1.x
  --v2                  Use TensorFlow Lite Converter 2.x
  --graph_def           Use graph def file(default)
  --saved_model         Use saved model
  --keras_model         Use keras model
  -i INPUT_PATH, --input_path INPUT_PATH
                        Full filepath of the input file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Full filepath of the output file.
  -I INPUT_ARRAYS, --input_arrays INPUT_ARRAYS
                        Names of the input arrays, comma-separated.
  -s INPUT_SHAPES, --input_shapes INPUT_SHAPES
                        Shapes corresponding to --input_arrays, colon-
                        separated.(ex:"1,4,4,3:1,20,20,3")
  -O OUTPUT_ARRAYS, --output_arrays OUTPUT_ARRAYS
                        Names of the output arrays, comma-separated.

```
