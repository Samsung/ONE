## onnx_autosubgraphs
in this project, we can support onnx AI model auto subgraph, the AI model can be  splitted by model size, operators etc, it help much for on-device AI acceleration, and has been verified on Rose-P NPU and Qualcomm DSP.

### OS environment
     1. ubuntu >=20.04
     2. GCC >= 9.4.0
     3. cmake >= 3.10
     4. python >= 3.8
     5. apt-get install libprotobuf-dev protobuf-compiler

### Python packages
    onnx                         1.16.0
    onnxruntime                  1.18.1
    onnxsim                      0.4.36
    torch                        2.3.1

### Pre-steps
    1. prepare the target onnx AI model, we use test.onnx for example
    2. use onnxsim to remove the complex structures before excution onnx-subgraph
    
### building the onnx-subgraph
    1. cd onnx-subgraph
    2. mkdir build & cd build
    3. cmake .. & make
    4. onnx-subgraph will be generated at ./build
    
### Parse the onnx model
    1. edit the config.json as your needs
       -> NPU_supported_ops mean operators that can be supported by NPU
       -> CPU_supported_ops mean operators that can be supported by CPU
       -> In case of operators supported by both CPU and NPU, we can describ the performance data at "performance_data"
       -> "max_subgraph_size": can set the max size of subgraph, it works only if NPU_supported_ops is NULL

    2. ./onnx-subgraph --onnx=test.onnx
       after parse done, subgraphs_ios.txt will be generated
       
 ### Split the onnx model to subgraphs
    1. edit the config path and model file path at extract_onnx.py 

    2. python extract_onnx.py, after extraction done, the subgraphs will be saved at './subgraphs'
    
### Verify the subgraphs inference with original model file
    1. edit the model path, subgraph path and config path in single_vs_multiple_onnx.py

    2. edit the input shape and name of onnx model in single_vs_multiple_onnx.py

    3. compare the MSE of original inference result and subgraphs inference result
       python single_vs_multiple_onnx.py
