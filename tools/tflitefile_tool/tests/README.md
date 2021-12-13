# How to test

## Prepare

There is `convolution_test.tflite` in `ONE/tests/scripts/models/cache/convolution_test.tflite`.

```
ONE$ find ./tests/scripts/models/ -name "convolution_test.tflite"
./tests/scripts/models/cache/convolution_test.tflite
```

## Test

```
ONE/tools/tflitefile_tool$ python -m unittest discover

----------------------------------------------------------------------
Ran 1 tests in 0.000s

OK
```

OR

```
ONE/tools/tflitefile_tool$ python ./tests/main.py

----------------------------------------------------------------------
Ran 1 tests in 0.000s

OK
```

## Reference

https://docs.python.org/3.6/library/unittest.html
