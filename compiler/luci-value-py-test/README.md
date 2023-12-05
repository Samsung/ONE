# luci-value-py-test

`luci-value-py-test` validates luci IR graph model file (.circle)

The test proceeds as follows

Step 1: Generate tflite files and circle files from TFLite recipes (listsed in test.lst).
```
"TFLite recipe" -> tflchef -> "tflite file" -> tflite2circle -> "circle file"
```

Step 2: Run TFLite interpreter and luci-interpreter for the generated tflite and circle, respectively.
(with the same input tensors filled with random values)
```
circle file -> luci-interpreter -------> Execution result 1
tflite file -> TFLite interpreter -----> Execution result 2
```

Step 3: Compare the execution result 1 and 2. The result must be the same.
