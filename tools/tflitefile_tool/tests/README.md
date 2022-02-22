# How to test

## Prepare

There is `add.tflite` in `ONE/nnpackage/examples/v1.0.0/add`.

```
ONE$ find ./nnpackage -name "add.tflite"
./nnpackage/examples/v1.0.0/add/add.tflite
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
