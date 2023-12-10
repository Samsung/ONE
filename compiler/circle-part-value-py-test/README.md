# circle-part-value-py-test

_circle-part-value-py-test_ evaluates partitioned models produced by circle-partitioner.

### Process of evaluation

Evaluation process is like how _luci-value-test_ does.

1) generates random input and stores to reference input file(s)
2) executes tflite file from common-artifacts for reference output
3) partitions circle file with .part file and produces into output folder
4) executes produced partitioned circle models with reference input file(s)
5) saves output(s) of circle models to file(s)
6) compares reference output with saved output file(s)
7) fail test if values differ
