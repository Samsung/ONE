# tflite2circle

_tflite2circle_ is a Tensorflow Lite to Circle model converter.

## Usage

Provide _tflite_ file input path and _circle_ file output path as a parameter to convert.

```
$ tflite2circle in.tflite out.circle
```

### --replace-unsupported-with-custom

If this option is enabled, any TFLite operation that is not currently supported by
 Circle will be automatically replaced with a CustomOp.

This allows you to convert and test models even if some operations are not yet 
implemented in the Circle.
