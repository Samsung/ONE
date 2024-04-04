# luci-ref-value-py-test

`luci-ref-value-py-test` validates luci IR graph model file (.circle) that current
`luci-value-py-test` cannot validate, such as tflite models with unsupported
data types like `INT4`.

The test proceeds as follows:

Step 1: Use tflite file from common-artifacts and generate circle files from tflite file
(listsed in test.lst).
```
"tflite file" -> tflite2circle -> "circle file"
```

Step 2: Read reference input files and run luci-eval-driver and get output files.
```
"circle file" -> luci-evel-driver -> "execution result"
```

Step 3: Compare with reference output files with the execution result. The result must be the same.

Reference input/output files are text files with simple fixed format like follows.
- first line is shape like `1,4`
- second line is data type like `float32`
- third line is values like `0.1,0.2,0.3,0.4`

Place them in same folder where `test.recipe` file exist, with names `ref.inputN` and `ref.outputN`,
where `N` is I/O number from `0`.
