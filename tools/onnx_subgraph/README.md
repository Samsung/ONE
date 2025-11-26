# onnx_autosubgraph
onnx-subgraph tool provides model auto partitionioning of onnx model to several sub models by
operator, performance and model size limitations, with the order and input / output names of
sub models.

# How to build the onnx-subgraph
## OS environment dependence
     1. ubuntu >=20.04
     2. GCC >= 9.4.0
     3. cmake >= 3.10
     4. python >= 3.8
     5. apt-get install libprotobuf-dev protobuf-compiler libjsoncpp-dev

## Python packages dependence
    onnx                         1.16.0
    onnxruntime                  1.18.1
    onnxsim                      0.4.36
    torch                        2.3.1
    scikit-image
    scikit-learn
    pandas
    tqdm

## building the onnx-subgraph
```bash
    cd onnx-subgraph
    mkdir build & cd build
    cmake .. & make
```
    we can get following output at './build'
```bash
    scripts
    ├── extract_onnx.py
    └── test_model_download.sh
    └── subgraphs_ios.txt
```

# How to use the onnx-subgraph
## Pre-steps
### Download the test AI models
    1. 'bash scripts/test_model_download.sh', then "resnet-test.onnx" will be got in './build'
    2. you can change to any other onnx files as your needs, or edit the download link in
       "scripts/test_model_download.sh"

## Parse the onnx model
    note: 'subgraphs_ios.txt' will be generated in future code, suppose we already have it as
    the example file now.

## Split the onnx model to subgraphs
```bash
    python scripts/extract_onnx.py \
              -s ./scripts/subgraphs_ios.txt \
              -m ./resnet-test.onnx
```
    after extraction done, the subgraphs will be saved at './subgraphs'
```bash
    subgraphs
    ├── CPUsubgraph0.onnx
    └── CPUsubgraph1.onnx
    ├── NPUsubgraph0.onnx
    └── NPUsubgraph1.onnx
```
## Verify the subgraphs inference with original model file
compare the MSE of original inference result and subgraphs inference result
```bash
    python scripts/single_vs_multiple_onnx.py \
           -s ./resnet-test.onnx \
           -m ./subgraphs/ \
           -n scripts/subgraphs_ios.txt
```
output:
```bash
    Single model inference completed!
    Multiple subgraph inference completed!
    Comparing inference results between single ONNX model and multiple subgraphs...
    Output '316' MSE: 5.125894080395578e-14
```
