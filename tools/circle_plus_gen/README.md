# Circle+ generator

Circle+ is a circle file which contains training hyperparameters. <br/> 
This tool generates a circle+ by adding training hyperparameters to a circle file.<br/>
It also helps handle circle+ file, such as checking whether the circle file contains training hyperparameters. <br/> 

## Requirements

This tool tested on python3.8. 

1. (optional) Set python virtaul environment.

    ```
    python3 -m venv venv
    source ./venv/bin/activate
    ```

2. Install required pakcages. 

    Currently, only `flatbuffers` is needed.
    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Add training hyperparameters using json file

You can add a training hyperparameters to a circle file.
To begin with, you need to write the hyperparameters in a json file. Here's [an example](./example/train_tparam.json) of a json file.

```bash 
cat example/train_tparam.json

# {
#   "optimizer": {
#       "type": "adam",
#       "args": {
#           "learningRate": 0.01,
#           "beta1": 0.9,
#           "beta2": 0.999,
#           "epsilon": 1e-07
#       }
#   },
#   "loss": {
#       "type": "categorical crossentropy",
#       "args": {
#           "fromLogits": true,
#           "reduction": "sum over batch size"
#       }
#   },
#   "batchSize": 32
# }
```

Next, execute the `main.py` script to add the hyperparameters to the `*.circle` file.

```bash
python3 main.py example/sample.circle example/train_tparam.json out.circle

# expected output
# 
# load training hyperparameters
# {
#     "optimizer": {
#         "type": "adam",
#         "args": {
#          ... 
#     },
#     "batchSize": 32
# }
# succesfully add hyperparameters to the circle file
# saved in out.circle
```

If you don't give `out.circle` as an argument, the input circle file(here, `example/sample.circle`) will be overwritten. 


## Print training hyperparameters in circle file

You can check whether the circle file contains the training hyperparameters.</br>
If you run the `main.py` without providing a json file, it will display training hyperparameters in the given circle file.

Try this with the files in [example](./example/).
```bash
python3 main.py example/sample.circle

# expected output
#
# check hyperparameters in example/sample.circle
# No hyperparameters
```
```bash
python3 main.py example/sample_tparam.circle

# expected output 
#
# check hyperparameters in example/sample_tparam.circle
# {
#     "optimizer": {
#         "type": "sgd",
#         "args": {
#             "learningRate": 0.0010000000474974513
#         }
#     },
#     "loss": {
#         "type": "sparse categorical crossentropy",
#         "args": {
#             "fromLogits": true,
#             "reduction": "SumOverBatchSize"
#         }
#     },
#     "batchSize": 64
# }
```

If it doesn't work well with example files, please check their md5sum to make sure they're not broken. 

```bash
$ md5sum example/sample.circle example/sample_tparam.circle

df287dea52cf5bf16bc9dc720e8bca04  example/sample.circle
6e736e0544acc7ccb727cbc8f77add94  example/sample_tparam.circle
```
