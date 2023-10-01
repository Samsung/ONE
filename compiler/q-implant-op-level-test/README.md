# q-implant-op-level-test

`q-implant-op-level-test` validates that q-implant supports common used operators.

The test proceeds as follows

Step 1: Generate tflite files and circle files from TFLite recipes (listsed in test.lst).
```
"TFLite recipe" -> tflchef -> "tflite file" -> tflite2circle -> "circle file"
```

Step 2: Generate qparam file(.json) and numpy array(.npy) for the operator python file.
```
operator file -> qparam file, numpy array
```

Step 3: Generate output.circle to use q-implant
```
"circle file" + "qparam.json" -> q-implant -> "quant circle file"
```
