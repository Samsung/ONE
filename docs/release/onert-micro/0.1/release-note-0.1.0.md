## Release Notes for onert-micro 0.1.0

_onert-micro_ is tiny runtime specialized for running NN model in MCU boards. Note that onert-micro is under active development and is subject to change.

### Supported operations

For MCU board, we support 22 operations as follows : 

   ADD, FULLY_CONNECTED, CONV_2D, LOGISTIC ,GATHER, EXPAND_DIMS, PACK, RESHAPE, REDUCE_PROD, LESS, MUL, MAX_POOL_2D, CONCATENATION, SHAPE, SLICE, SUB, SPLIT, STRIDED_SLICE, TANH, SOFTMAX, WHILE, UNIDIRECTIONAL_SEQUENCE_LSTM  

### RNN Model

#### LSTM

onert-micro supports Keras model with LSTM operations. But, it should be converted to UNIDIRECTIONAL_SEQUENCE_LSTM operation in circle format.

#### GRU 

onert-micro supports model with GRU Operations, which is converted from Keras Model. Please refer to https://github.com/Samsung/ONE/issues/10465 to see GRU operation supported by onert-micro. 

### Benchmark

onert-micro shows better performance than tflite-micro especially in memory consumption, binary size.

The measurement is done on TizenRT running reference models on the development board with the following spec : 

- 32-bit Arm Cortex-M33 200MHz
- 4MB RAM, 8MB Flash

Commit for measurement : 
- tflite-micro commit: https://github.com/tensorflow/tflite-micro/commit/4e62ea7b821c1e6af004912132395fb81922ea8d

- onert-micro commit: https://github.com/Samsung/ONE/commit/c763867500fe3d80bfd1ef834990d34a81640d17
#### L model

| Params                            | Tflite micro    | Onert-micro |
|-----------------------------------|---------------|-------------|
| Execution time(us)*               | **2 912 700** | 2 953 000   |
| RAM consumption(bytes)            | 126 800       | **93 376**  |
| Binary file size overhead (bytes) | 57 676        | **32 248**  |


#### T1 model

Params | Tflite micro | Onert-micro |
--- | --- | --- 
Execution time(us)* | **1 340** | 1 510 | 
RAM consumption(bytes) | 1 640 | **1 152** |
Binary file size overhead (bytes) | 35 040 | **19 432** |

#### T2 model

Params | Tflite micro** | Onert-micro |
--- | --- | --- 
Execution time(us)* | N/A | 5 090 | 
RAM consumption(bytes) | N/A | 3 360 |
Binary file size overhead (bytes) | N/A | 30 488 |

#### Model with GRU operations

- model link : https://github.com/Samsung/ONE/files/8368702/gru.zip

Params | Tflite micro** | Onert-micro |
--- | --- | --- 
Execution time(us)* | N/A | 335 000 | 
RAM consumption(bytes) | N/A | 14 816 |
Binary file size overhead (bytes) | N/A | 43 444 |


(*) Average for 100 inferences
(**) Tflite-micro has not launched this model

