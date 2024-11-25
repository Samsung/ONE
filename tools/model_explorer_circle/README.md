# Circle adapter for Model Explorer

Circle adapter is an extension adapter for displaying a Circle model in Model Explorer.

* [Model Explorer](https://github.com/google-ai-edge/model-explorer)
  * Visulization tool for various type of model graphs such as onnx, tflite, pt2 and mlir.

## Prerequisites
```
python -m venv venv
source ./venv/bin/activate
```

## Installation
```
pip install -e .
```

## How to use
```
model-explorer --extension circle_adapter --no_open_in_browser
```

## How to test
```
pip intall pytest
pytest
```
