# Supported Operations

As of 2020-06-26
- TensorFlow commit e5023a1738cce7efcdf9d87863b85c80ab2f8c9e
- This commit is Tensorflow nightly build after v2.2.0 release

TensorFlow lite operators | circle IR | Compiler | Runtime
-- | -- | -- | --
ABS | O | O | O
ADD | O | O | O
ADD_N | O | O |  
ARG_MAX | O | O | O
ARG_MIN | O | O |  
AVERAGE_POOL_2D | O | O | O
BATCH_MATMUL | O | O | O
BATCH_TO_SPACE_ND | O | O |  
BIDIRECTIONAL_SEQUENCE_LSTM | O |   |  
BIDIRECTIONAL_SEQUENCE_RNN | O |   |  
CALL | O |   |  
CAST | O | O | O
CEIL | O | O |  
CONCAT_EMBEDDINGS | O |   |  
CONCATENATION | O | O | O
CONV_2D | O | O | O
COS | O | O | O
CUSTOM | O | O | O
DELEGATE | O |   |  
DENSIFY | O |   |  
DEPTH_TO_SPACE | O | O | O
DEPTHWISE_CONV_2D | O | O | O
DEQUANTIZE | O |   | O
DIV | O | O | O
ELU | O | O |  
EMBEDDING_LOOKUP | O |   | O
EMBEDDING_LOOKUP_SPARSE | O |   |  
EQUAL | O | O | O
EXP | O | O | O
EXPAND_DIMS | O | O | O
FAKE_QUANT | O |   |  
FILL | O | O | O
FLOOR | O | O | O
FLOOR_DIV | O | O |  
FLOOR_MOD | O | O |  
FULLY_CONNECTED | O | O | O
GATHER | O | O | O
GATHER_ND | O | O |  
GREATER | O | O | O
GREATER_EQUAL | O | O | O
HARD_SWISH | O |   |  
HASHTABLE_LOOKUP | O |   | O
IF | O | O | O
L2_NORMALIZATION | O | O | O
L2_POOL_2D | O | O | O
LEAKY_RELU | O | O |  
LESS | O | O | O
LESS_EQUAL | O | O | O
LOCAL_RESPONSE_NORMALIZATION | O | O | O
LOG | O | O | O
LOG_SOFTMAX | O | O |  
LOGICAL_AND | O | O | O
LOGICAL_NOT | O | O | O
LOGICAL_OR | O | O | O
LOGISTIC | O | O | O
LSH_PROJECTION | O |   |  
LSTM | O |   | O
MATRIX_DIAG | O |   |  
MATRIX_SET_DIAG | O |   |  
MAX_POOL_2D | O | O | O
MAXIMUM | O | O | O
MEAN | O | O | O
MINIMUM | O | O | O
MIRROR_PAD | O | O |  
MUL | O | O | O
NEG | O | O | O
NON_MAX_SUPPRESSION_V4 | O |   |  
NON_MAX_SUPPRESSION_V5 | O |   |  
NOT_EQUAL | O | O | O
ONE_HOT | O | O | O
PACK | O | O | O
PAD | O | O | O
PADV2 | O |   |  
POW | O | O | O
PRELU | O | O | O
QUANTIZE | O |   |  
RANGE | O | O | O
RANK | O | O |  
REDUCE_ANY | O | O | O
REDUCE_MAX | O | O | O
REDUCE_MIN | O | O | O
REDUCE_PROD | O | O | O
RELU | O | O | O
RELU_N1_TO_1 | O | O |  
RELU6 | O |   | O
RESHAPE | O | O | O
RESIZE_BILINEAR | O | O | O
RESIZE_NEAREST_NEIGHBOR | O | O |  
REVERSE_SEQUENCE | O | O |  
REVERSE_V2 | O |   | O
RNN | O |   | O
ROUND | O | O | O
RSQRT | O | O | O
SCATTER_ND | O | O |  
SEGMENT_SUM | O | O |  
SELECT | O | O | O
SELECT_V2 | O |   |  
SHAPE | O | O | O
SIN | O | O | O
SKIP_GRAM | O |   |  
SLICE | O | O |  
SOFTMAX | O | O | O
SPACE_TO_BATCH_ND | O | O | O
SPACE_TO_DEPTH | O | O | O
SPARSE_TO_DENSE | O | O |  
SPLIT | O | O | O
SPLIT_V | O | O |  
SQRT | O | O | O
SQUARE | O | O |  
SQUARED_DIFFERENCE | O | O | O
SQUEEZE | O | O | O
STRIDED_SLICE | O | O | O
SUB | O | O | O
SUM | O | O | O
SVDF | O |   |  
TANH | O | O | O
TILE | O | O | O
TOPK_V2 | O | O | O
TRANSPOSE | O | O | O
TRANSPOSE_CONV | O | O | O
UNIDIRECTIONAL_SEQUENCE_LSTM | O |   |  
UNIDIRECTIONAL_SEQUENCE_RNN | O |   |  
UNIQUE | O |   |  
UNPACK | O | O | O
WHERE | O |   |  
WHILE | O | O |  
ZEROS_LIKE | O | O |  
Count | 127 | 97 | 81


### circle additional operators

Operator | compiler | runtime
-- | -- | --
BCQ_FULLY_CONNECTED | O | O
BCQ_GATHER | O | O
INSTANCE_NORM | O | O
