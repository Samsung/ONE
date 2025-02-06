# onnx_autosubgraph
onnx-subgraph tool provides  model auto partitionioning of onnx model to several sub models by 
operator, performance and model size limitations,with the order and input / output names of 
sub models

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
    1. cd onnx-subgraph
    2. mkdir build & cd build
    3. cmake .. & make
    4. we can get following output at ./build
          └── scripts
              ├── extract_onnx.py
              └── test_model_download.sh
              └── subgraphs_ios.txt

# How to use the onnx-subgraph
## Pre-steps
### Download the test AI models
    1. bash scripts/test_model_download.sh, then "resnet-test.onnx" will be got in ./build
    2. you can change to any other onnx files as your needs, or edit the download link in 
	   "scripts/test_model_download.sh"
  
## Parse the onnx model
    note: subgraphs_ios.txt will be generated in future code, suppose we already have it as 
    the example file now
       
## Split the onnx model to subgraphs
    1. edit the config path and model file path at ./scripts/extract_onnx.py 
       e.g.: extract_onnx_lib.split_onnx_ios('./scripts/subgraphs_ios.txt','./resnet-test.onnx') 
    2. python scripts/extract_onnx.py, after extraction done, the subgraphs will be saved 
	   at './subgraphs'
       subgraphs
       ├── CPU
       │   ├── CPUsubgraph0.onnx
       │   └── CPUsubgraph1.onnx
       └── NPU
           ├── NPUsubgraph0.onnx
           └── NPUsubgraph1.onnx
