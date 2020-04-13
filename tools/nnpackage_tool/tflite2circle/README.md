# tflite2circle

`tflite2circle` is a tool to convert tflite into circle.

## Usage

```
Usage: tflite2circle.sh [options] tflite
Convert tflite to circle

Returns
     0       success
  non-zero   failure

Options:
    -h   show this help
    -o   set output directory (default=.)

Environment variables:
   flatc           path to flatc
                   (default=./build/externals/FLATBUFFERS/build/flatc)
   tflite_schema   path to schema.fbs
                   (default=./externals/TENSORFLOW-1.12/tensorflow/contrib/lite/schema/schema.fbs)

Examples:
    tflite2circle.sh Add_000.tflite         => convert Add_000.tflite into Add_000.circle
    tflite2circle.sh -o my/circles Add_000  => convert Add_000.tflite into my/circles/Add_000.circle
```
