# dredd-rule-lib

*dredd-rule-lib* is a library that defines functions to run *dredd* tests, which checks non-functional aspect of compiled files.

## Terms

Assume that we want to check the size of generated tflite file to be less than 1024 Bytes.
In such case, we'd like to use the following terms:

- "metric" : *file size*
- "rule" : *file size < 1024*
- "metric function": `file_size` that returns size of a compiled tflite file

Models (input of test) exist in *model repo*, where

- "model repo" : directory where models exist. For *tf2tflite-dredd-pbtxt-test*, model repo is
  `res/TensorFlowTests`.

## Metrics supported

The following metric functions are provided:
- `all_op_count` : the count of operations inside a compiled tflite file
- `file_size` : the size of compiled tflite file
- `tensor_shape` : The shape of a specific node in a compiled tflite file. The format looks like `[1,-1,7,2]`(without spaces).
- In addition, `op_count`, `conv2d_weight_not_constant`, etc.
- Please , refer to [`rule-lib.sh`](rule-lib.sh) for metric functions

## Related projects - *dredd* tests

Four *dredd* test projects use *dredd-rule-lib*:

- *tf2tflite-dredd-pbtxt-test*
  - Models in `pbtxt`, text file, are compiled into `tflite` file.
  - Then `rule` file that each model has is checked against the `tflite` file.
- *tf2tflite-dredd-pb-test*
  - Models in `pb`, binary file, are compiled into `tflite` file.
  - Then `rule` file that each model has is checked against the `tflite` file.
- *tf2circle-dredd-pbtxt-test*
  - Models in `pbtxt`, text file, are compiled into `circle` file.
  - Then `rule` file that each model has is checked against the `circle` file.
- *tf2circle-dredd-pb-test*
  - Models in `pb`, binary file, are compiled into `circle` file.
  - Then `rule` file that each model has is checked against the `circle` file.

## Rule file

To be a target of *dredd*-tests, a `.rule` file **must** exist in a model directory.
Please refer to `res/TensorFlowTests/NET_0025/tflite_1.0_rel_requirement.rule` for an example.

### Naming convention of rule file

Note that the file name `tflite_1.0_rel_requirement.rule` is our convention containing the
information below:
- Generated file type (`tflite`)
- SDK version (`1.0_rel`)
- Purpose (`requirement`)

## How do all these work?

For *tf2tflite-dredd-pbtxt-test*, (*tf2circle-dredd-pbtxt-test* works similarly)

```
model repo                                   tf2tflite-dredd-pbtxt-test
-----------------------------------------------------------------------------------------------
   NET_0025
    ├── test.pbtxt  ---------------------->  converted to NET_0025.pb, and then NET_0025.tflite
    |                                       /|\
    ├── test.info ---------------------------+
    |   (input/output info of model)
    |
    └── tflite_1.0_rel_requirement.rule -->  running rule file against tflite --> pass or fail
                                                      /|\
                          dredd-rule-lib               | (using)
                      ----------------------           |
                          rule-lib.sh                  |
                            - defining rule function --+
```

For *tf2tflite-dredd-pb-test*, (*tf2circle-dredd-pb-test* works similarly)

```
model repo                                   tf2tflite-dredd-pb-test
-----------------------------------------------------------------------------------------------
   Inception_v3
    ├── model.pb  ------------------------>  converted to Inception_v3.tflite
    |                                       /|\
    ├── model.info --------------------------+
    |   (input/output info of model)
    |
    └── tflite_1.0_rel_requirement.rule -->  running rule file against tflite --> pass or fail
                                                      /|\
                          dredd-rule-lib               | (using)
                      ----------------------           |
                          rule-lib.sh                  |
                            - defining rule function --+
```

## Model repo and How to add a model as a target of a *dredd*-test.

For *tf2tflite-dredd-pbtxt-test* and *tf2circle-dredd-pbtxt-test*,
model repo is `res/TensorFlowTests`.

To add a model into these tests, the model directory name should be added into one of the following files:
- `test.lst` : This file resides in git
- `test.local.lst` : This file is ignored by git. Use this for personal purpose.

For *tf2tflite-dredd-pb-test* and *tf2circle-dredd-pb-test*,
model repo is `tf2tflite-dredd-pb-test/contrib` and .`tf2circle-dredd-pb-test/contrib` respectively.

Use these tests for binary models in large size.

To add a model into these tests, the model directory name should be added into the following file:
- `contrib.lst` : This file is ignored by git.
