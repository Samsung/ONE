# Circle+ generator

If a circle file has training hyperparameters, we usually call it a 'circle+' file.<br/>
This tool generates a circle+ file by injecting training hyperparameters into a circle file.<br/>
It also helps handle circle+ file, such as checking whether the circle file contains training hyperparameters. <br/> 

## Requirements

1. (optional) Set python virtaul environment.

    This tool tested on python3.8.

    ```
    python3 -m venv venv
    source /venv/bin/activate
    ```

2. Install required pakcages. 

    Currently, only `flatbuffers==24.3.25` is needed.
    ```bash
    pip install -r requirements.txt
    ```

## Inject training parameters using json file

<!--to be updated -->

## Check if the circle file contains training parameters

You can check whether the circle file contains the training parameters.</br>
If you run the `main.py` without providing a json file, it will check training parameters and display them.

Try this with the files in [example](./example/).
```bash
python3 main.py example/mnist.circle
```
```bash
python3 main.py example/mnist_with_tparam.circle
```
