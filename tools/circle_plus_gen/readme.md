
# Circle+ generator

If a circle file has training hyperparameters, we usually call it a 'circle+' file.<br/>
This tool generates a circle+ file by injecting training hyperparameters into a circle file.

## Requirements

1. (optional) Set up a python virtaul environment.
```
python -m venv venv
source /venv/bin/activate
```

2. Install the required pakcages. 

Currently, only `flatbuffers==24.3.25` is needed.
```bash
pip install -r requirements.txt
```


## Inject training parameters using json file

You can use this tool to inject a training hyperparameters into a circle file.\
To begin with, you need to write the hyperparameters in a json file. Here's an [example](./example/train_parameter.json) of a json file.


```bash 
cat example/train_parameter.json

# {
#     "optimizer": {
#         "type": "Adam",
#         "args": {
#             "learningRate": 0.01,
#             "beta1": 0.9,
#             "beta2": 0.999,
#             "epsilon": 1e-07
#         }
#     },
#     "loss": {
#         "type": "CategoricalCrossentropy",
#         "args": {
#             "fromLogits": true,
#             "reduction": "SumOverBatchSize"
#         }
#     },
#     "batchSize": 32
# }
``` 

Next, execute the `main.py` script to inject `*.json` file to the `*.circle` file. 
```bash 
python3 main.py example/mnist.circle example/train_parameter.json --v
```

### Check if the circle file has training parameters

You can check whether the circle file contains the training parameters.</br>
If you run the `main.py` without providing a json file, it will check if the circle file has training parameters and display them.

```bash
python3 main.py example/mnist.circle
```
