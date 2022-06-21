## Introduce post-process stage for one-infer

### Summary

This draft is aiming to introduce post-process stage by option in `one-infer`.

### Motivation

To support *value-test* between two different `one-infer` results, the inference result data should be well formatted.
The standard format of data was decided to use `h5` because `h5` is a type which can contain multiple data with specific hierarchy and some other reasons. (There is a discussion in [this link](https://github.com/Samsung/ONE/issues/9248))

On the other hand, `one-infer` cannot guarantee various backend drivers emits well defined format data. 

So, `one-infer` needs to solve that problem with `post-process` option which run data conversion after driver execution.

### How it works

```console
$ ls
model.npu aaa.0.npy aaa.1.npy

# Note that [commands] run model.npu with aaa.0.npy and aaa.1.npy as input data 
# and emit output data as bbb.0.npy ...
# ./npy2h5.py can be any script which converts the npu-infer's result 
# to preset standard format of h5
$ one-infer -d npu-infer --post-process ./npy2h5.py -- [commands]
Inference complete!

$ ls
model.npu aaa.0.npy aaa.1.npy bbb.0.npy bbb.1.npy ccc.h5
```

