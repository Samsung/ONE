# Circle-OpSelector-test

## Usage

- This test doesn't consider **if** and **while** nodes yet. 

**Single file test**

- `tflite-file-name` must be contained in [tflite receipes](https://github.com/Samsung/ONE/tree/5e53631d84a05467547296dd2f948b8a7e907161/res/TensorFlowLiteRecipes)
- `selected nodes` format is `"0 1 3"`

```
$ ./opselector_test.sh [tflite-file-name] [selected nodes]
# ex)
$ ./compiler/circle-opselector-test/opselector_test.sh Part_Sqrt_Rsqrt_Add_002 "0 1 2 3"
```

This test compares between the result of `circle-opselector` and [select-operator.py](https://github.com/Samsung/ONE/blob/master/tools/tflitefile_tool/select_operator.py) in tflite_tool.

The detailed process is as follows

1. Prepare a `tflite` model.
2. Create a `circle` version of that `tflite` model using 'tflite2circle`.
3. Create a `tflite` subgraph using `select_operator.py`.
4. Create a `circle` subgraph using `Circle-OpSelector`.
5. Run `nnpckgtc` with input as the `tflite` subgraph to get the golden value and the package.
6. Modify the **package** so that it contains the `circle` subgraph, not the `tflite` subgraph. This step is due to the fact that the circle model can't be an input of `nnpckgtc`.
7. Run the **package** in **runtime** to get the output of the `circle` subgraph package, with the same input used for getting the golden value.
8. Compare the golden value with the **runtime** output.



**Multiple file test**

- `test.lst` : Write tflite file names you want to test
- This test only covers **continuous selecting cases**

```Shell
python ./opselector_test.py
```



**CLI Option test**

- This test is checking cli options roughly.

- Use only one tflite file(Part_Sqrt_Rsqrt_002.circle)

```Shell
python ./cli_test.py
```
