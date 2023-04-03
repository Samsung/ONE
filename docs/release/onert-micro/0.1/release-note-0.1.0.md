## Release Notes for onert-micro 0.1.0

_onert-micro_ is tiny runtime specialized for running NN model in MCU boards. Note that onert-micro is under active development and is likely to change.

### Supported operations

For MCU board, we support 18 operations as follows : 

   ADD, FULLY_CONNECTED, CONV_2D, LOGISTIC ,GATHER, EXPAND_DIMS, RESHAPE, LESS, MUL, MAX_POOL_2D, CONCATENATION, SLICE, SUB, SPLIT, TANH, SOFTMAX, WHILE, UNIDIRECTIONAL_SEQUENCE_LSTM  

### GRU Model 

onert-micro supports GRU model, which is converted from Keras Model. Please refer to https://github.com/Samsung/ONE/issues/10465 about GRU model in Keras. 
