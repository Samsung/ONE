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

    Currently, only `flatbuffers==24.3.25` is needed.
    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Inject training hpyerparameters using json file

<!--to be updated -->

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
# check hyperparameters in example/sample_tpram.circle
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
$ md5sum example/sample.circle example/sample_tpram.circle

df287dea52cf5bf16bc9dc720e8bca04  example/sample.circle
e8c737488ce3ab1b60d4fd15dea770c8  example/sample_tpram.circle
```
