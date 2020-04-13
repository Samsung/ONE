# tfl-verify

_tfl-verify_ allows users to verify TF Lite models.

## Usage

Provide _tflite_ file as a parameter to verify validity.

```
$ tfl-verify tflitefile.tflite
```

Result for valid file
```
[ RUN       ] Check tflitefile.tflite
[      PASS ] Check tflitefile.tflite
```

Result for invalid file
```
[ RUN       ] Check tflitefile.tflite
[      FAIL ] Check tflitefile.tflite
```
