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
    2. bash 3rd_files_download.sh
    3. mkdir build & cd build
    4. cmake .. & make
    5. we can get following output at ./build
          ├── onnx-subgraph
          └── scripts
              ├── config.json
              ├── config-sample-1.json
              ├── config-sample-2.json
              ├── extract_onnx_lib.py
              ├── extract_onnx.py
              ├── model_inference_multiple_output.py
              ├── model_inference.py
              ├── onnx_subgraph_ut.py
              ├── quant.py
              ├── single_vs_multiple_onnx.py
              └── test_model_download.sh
# How to use the onnx-subgraph
## Pre-steps
### Download the test AI models
    1. bash scripts/test_model_download.sh, then "resnet-test.onnx" will be got in ./build
    2. you can change to any other onnx files as your needs, or edit the download link in 
	   "scripts/test_model_download.sh"
### Prepare the config.json
    1. edit the config.json
       . you can edit operators in "NPU_supported_ops" and "CPU_supported_ops";
       . you can edit performance data in "performance_data" as the real HW status, 
       . you can edit "max_subgraph_size" in case of "NPU_supported_ops" is []
    2. you can also check more examples in "config-sample-1.json" and "config-sample-2.json"

  
## Parse the onnx model
     ./onnx-subgraph --onnx=resnet-test.onnx
       after parsing done, subgraphs_ios.txt will be generated at current path
       
## Split the onnx model to subgraphs
    1. edit the config path and model file path at ./scripts/extract_onnx.py 
       e.g.: extract_onnx_lib.split_onnx_ios('./subgraphs_ios.txt','./resnet-test.onnx') 
    2. python scripts/extract_onnx.py, after extraction done, the subgraphs will be saved 
	   at './subgraphs'
       subgraphs
       ├── CPU
       │   ├── CPUsubgraph0.onnx
       │   └── CPUsubgraph1.onnx
       └── NPU
           ├── NPUsubgraph0.onnx
           └── NPUsubgraph1.onnx
    
## Verify the subgraphs inference with original model file
    1. edit the model path, subgraph path and config path in ./scripts/single_vs_multiple_onnx.py
             single_onnx_model_path = './resnet-test.onnx'
             model_path = './subgraphs/'
             subgraphsiostxt_path = './subgraphs_ios.txt'
    2. edit the input shape and name of onnx model in ./scripts/single_vs_multiple_onnx.py
             default_input_data = {
                 "x": np.random.rand(1, 3, 256, 256).astype(np.float32),
             }
    3. compare the MSE of original inference result and subgraphs inference result
       python ./scripts/single_vs_multiple_onnx.py
       output:
            Single model inference completed!
            Multiple subgraph inference completed!
            Comparing inference results between single ONNX model and multiple subgraphs...
            Output '316' MSE: 5.125894080395578e-14
