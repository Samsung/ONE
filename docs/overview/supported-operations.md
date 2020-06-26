# Supported Operations

As of 2020-06-26
- TensorFlow commit e5023a1738cce7efcdf9d87863b85c80ab2f8c9e

TensorFlow lite operators | circle IR | Compiler | Runtime
-- | -- | -- | --
ABS | O | O |  
ADD | O | O |  
ADD_N | O | O |  
ARG_MAX | O | O |  
ARG_MIN | O | O |  
AVERAGE_POOL_2D | O | O |  
BATCH_MATMUL | O | O |  
BATCH_TO_SPACE_ND | O | O |  
BIDIRECTIONAL_SEQUENCE_LSTM | O |   |  
BIDIRECTIONAL_SEQUENCE_RNN | O |   |  
CALL | O |   |  
CAST | O | O |  
CEIL | O | O |  
CONCAT_EMBEDDINGS | O |   |  
CONCATENATION | O | O |  
CONV_2D | O | O |  
COS | O | O |  
CUSTOM | O | O |  
DELEGATE | O |   |  
DENSIFY | O |   |  
DEPTH_TO_SPACE | O | O |  
DEPTHWISE_CONV_2D | O | O |  
DEQUANTIZE | O |   |  
DIV | O | O |  
ELU | O | O |  
EMBEDDING_LOOKUP | O |   |  
EMBEDDING_LOOKUP_SPARSE | O |   |  
EQUAL | O | O |  
EXP | O | O |  
EXPAND_DIMS | O | O |  
FAKE_QUANT | O |   |  
FILL | O | O |  
FLOOR | O | O |  
FLOOR_DIV | O | O |  
FLOOR_MOD | O | O |  
FULLY_CONNECTED | O | O |  
GATHER | O | O |  
GATHER_ND | O | O |  
GREATER | O | O |  
GREATER_EQUAL | O | O |  
HARD_SWISH | O |   |  
HASHTABLE_LOOKUP | O |   |  
IF | O | O |  
L2_NORMALIZATION | O | O |  
L2_POOL_2D | O | O |  
LEAKY_RELU | O | O |  
LESS | O | O |  
LESS_EQUAL | O | O |  
LOCAL_RESPONSE_NORMALIZATION | O | O |  
LOG | O | O |  
LOG_SOFTMAX | O | O |  
LOGICAL_AND | O | O |  
LOGICAL_NOT | O | O |  
LOGICAL_OR | O | O |  
LOGISTIC | O | O |  
LSH_PROJECTION | O |   |  
LSTM | O |   |  
MATRIX_DIAG | O |   |  
MATRIX_SET_DIAG | O |   |  
MAX_POOL_2D | O | O |  
MAXIMUM | O | O |  
MEAN | O | O |  
MINIMUM | O | O |  
MIRROR_PAD | O | O |  
MUL | O | O |  
NEG | O | O |  
NON_MAX_SUPPRESSION_V4 | O |   |  
NON_MAX_SUPPRESSION_V5 | O |   |  
NOT_EQUAL | O | O |  
ONE_HOT | O | O |  
PACK | O | O |  
PAD | O | O |  
PADV2 | O |   |  
POW | O | O |  
PRELU | O | O |  
QUANTIZE | O |   |  
RANGE | O | O |  
RANK | O | O |  
REDUCE_ANY | O | O |  
REDUCE_MAX | O | O |  
REDUCE_MIN | O | O |  
REDUCE_PROD | O | O |  
RELU | O | O |  
RELU_N1_TO_1 | O | O |  
RELU6 | O |   |  
RESHAPE | O | O |  
RESIZE_BILINEAR | O | O |  
RESIZE_NEAREST_NEIGHBOR | O | O |  
REVERSE_SEQUENCE | O | O |  
REVERSE_V2 | O |   |  
RNN | O |   |  
ROUND | O | O |  
RSQRT | O | O |  
SCATTER_ND | O | O |  
SEGMENT_SUM | O | O |  
SELECT | O | O |  
SELECT_V2 | O |   |  
SHAPE | O | O |  
SIN | O | O |  
SKIP_GRAM | O |   |  
SLICE | O | O |  
SOFTMAX | O | O |  
SPACE_TO_BATCH_ND | O | O |  
SPACE_TO_DEPTH | O | O |  
SPARSE_TO_DENSE | O | O |  
SPLIT | O | O |  
SPLIT_V | O | O |  
SQRT | O | O |  
SQUARE | O | O |  
SQUARED_DIFFERENCE | O | O |  
SQUEEZE | O | O |  
STRIDED_SLICE | O | O |  
SUB | O | O |  
SUM | O | O |  
SVDF | O |   |  
TANH | O | O |  
TILE | O | O |  
TOPK_V2 | O | O |  
TRANSPOSE | O | O |  
TRANSPOSE_CONV | O | O |  
UNIDIRECTIONAL_SEQUENCE_LSTM | O |   |  
UNIDIRECTIONAL_SEQUENCE_RNN | O |   |  
UNIQUE | O |   |  
UNPACK | O | O |  
WHERE | O |   |  
WHILE | O | O |  
ZEROS_LIKE | O | O |  


### circle additional operators

Operator | compiler | runtime
-- | -- | --
BCQ_FULLY_CONNECTED | O | O
BCQ_GATHER | O | O
INSTANCE_NORM | O | O
