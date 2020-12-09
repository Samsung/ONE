/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file     NeuralNetworksEx.h
 * @brief    This file contains ANeuralNetworksModel_addOperationEx function definition
 * @note     This header describes experimental feature,
 *           so specification here can be changed or/and removed
 */
#ifndef NN_RUNTIME_NEURAL_NETWORKS_EX_H
#define NN_RUNTIME_NEURAL_NETWORKS_EX_H

#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * @brief Extended operation types
 */
typedef enum
{
  /** extends operation. */

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_CAST_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_CAST instead
   *
   */
  ANEURALNETWORKS_CAST_EX = 50000,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_GATHER_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_GATHER instead
   *
   */
  ANEURALNETWORKS_GATHER_EX = 50001, /**< Gather slices according to indexes and axis */

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_TOPK_V2_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_TOPK_V2 instead
   *
   */
  ANEURALNETWORKS_TOPK_V2_EX = 50002,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_REDUCE_MAX_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_REDUCE_MAX instead
   *
   */
  ANEURALNETWORKS_REDUCE_MAX_EX = 50003,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_SPLIT_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_SPLIT instead
   *
   */
  ANEURALNETWORKS_SPLIT_EX = 50004, /**< Splits a tensor into sub tensors */

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_RSQRT_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_RSQRT instead
   *
   */
  ANEURALNETWORKS_RSQRT_EX = 50005,

  /**
   * Computes element-wise squared difference on the input tensors.
   *
   * Takes two input tensors of identical {@link OperandCode} and compatible dimensions.
   * The output is the result of squaring of difference given by subtracting the second input tensor
   * from the first one.
   *
   * Two dimensions are compatible when:
   *     1. they are equal, or
   *     2. one of them is 1
   *
   * The size of the output is the maximum size along each dimension of the
   * input operands. It starts with the trailing dimensions, and works its way
   * forward.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: An n-D tensor, specifying the first input.
   * * 1: A tensor of the same {@link OperandCode}, and compatible dimensions
   *      as input0.
   *
   * Outputs:
   * * 0: The output tensor, of the same {@link OperandCode} as input0.
   */
  ANEURALNETWORKS_SQUARED_DIFFERENCE_EX = 50006,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_NEG_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_NEG instead
   *
   */
  ANEURALNETWORKS_NEG_EX = 50007,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_EXP_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_EXP instead
   *
   */
  ANEURALNETWORKS_EXP_EX = 50008,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_REDUCE_SUM_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_REDUCE_SUM instead
   *
   */
  ANEURALNETWORKS_REDUCE_SUM_EX = 50009,

  /**
   * A transposed convolutional layer carries out a regular convolution
   * but reverts its spatial transformation.
   * Transpose convolution basically performs convolution with transposed weights.
   *
   * Supported tensor {@link OperandCode}:
   * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: only 4
   *
   * Inputs:
   *   0: An {@link ANEURALNETWORKS_INT32} 1-D four element tensor, specifying the output shape.
   *   1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
   *      specifying the filter.
   *   2: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   *   3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the padding type.
   *   4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
   *      walking through input in the ‘width’ dimension.
   *   5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the stride when
   *      walking through input in the height dimension.
   *
   * Outputs:
   *   0: The output 4-D tensor, of shape [batches, out_height, out_width, depth_out].
   */
  ANEURALNETWORKS_TRANSPOSE_CONV_EX = 50010,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_EQUAL_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_EQUAL instead
   *
   */
  ANEURALNETWORKS_EQUAL_EX = 50011,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_ABS_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_ABS instead
   *
   */
  ANEURALNETWORKS_ABS_EX = 50012,
  /**
   * Packs a list of rank-R tensors into one rank- (R+1) tensor along the axis dimension.
   *
   * The input tensors must have identical {@link OperandCode} and the same
   * dimensions.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_INT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 3
   *
   * Inputs:
   * * 0 ~ n-1: The list of n input tensors, of shape
   *            [D0, D1, ..., Daxis(i), ..., Dm]. For inputs of
   *            {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, all input tensors
   *            must have the same scale and zeroPoint.
   * * n: An {@link ANEURALNETWORKS_INT32} scalar, specifying the
   *      number of input tensors.
   * * n+1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the
   *        pack axis.
   *
   * Outputs:
   * * 0: The output, a tensor of the same {@link OperandCode} as the input
   *      tensors. The output shape is [D0, D1, ..., N at Daxis(i), ..., Dm+1]
   *      where N is the number of tensors to be packed.
   */
  ANEURALNETWORKS_PACK_EX = 50013,
  /**
   * Unpacks a given rank-R tensors into num_splits rank- (R-1) tensors along the axis dimension.
   * num_splits has to respect integral divisibility of dimension value along axis dimension
   * of the input.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_INT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: The input shape is [D0, D1, ..., N at Daxis(i), ..., Dm+1].
   * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the
   *      number of splits along unpack axis.
   * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the
   *      unpack axis.
   *
   * Outputs:
   * * 0 ~ n-1: The list of n output tensors, of shape
   *            [D0, D1, ..., Daxis(i), ..., Dm]. The output tensors are of the same
   *            {@link OperandCode} as the input tensor 0.
   */
  ANEURALNETWORKS_UNPACK_EX = 50014,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_ARGMAX_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_ARGMAX instead
   *
   */
  ANEURALNETWORKS_ARGMAX_EX = 50015,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_SQRT_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_SQRT instead
   *
   */
  ANEURALNETWORKS_SQRT_EX = 50016,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_NOT_EQUAL_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_NOT_EQUAL instead
   *
   */
  ANEURALNETWORKS_NOT_EQUAL_EX = 50017,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_LOGICAL_NOT_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_LOGICAL_NOT instead
   *
   */
  ANEURALNETWORKS_LOGICAL_NOT_EX = 50018,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_LOGICAL_AND_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_LOGICAL_AND instead
   *
   */
  ANEURALNETWORKS_LOGICAL_AND_EX = 50019,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_LOGICAL_OR_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_LOGICAL_OR instead
   *
   */
  ANEURALNETWORKS_LOGICAL_OR_EX = 50020,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_REDUCE_MIN_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_REDUCE_MIN instead
   *
   */
  ANEURALNETWORKS_REDUCE_MIN_EX = 50021,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_PRELU_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_PRELU instead
   *
   */
  ANEURALNETWORKS_PRELU_EX = 50022,

  /**
   * Returns a one-hot tensor.
   *
   * The locations represented by indices in indices take value on_value, while all other locations
   * take value off_value.
   * The on_value and off_value must have matching data types. They must be the same data type as
   * specified by the data type of output.
   *
   * If the input indices is rank N, the output will have rank N+1. The new axis is created at
   * dimension axis.
   * If indices is a scalar the output shape will be a vector of length depth.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_INT32}
   *
   * Supported tensor rank: up to 4
   *
   * Supported tensor type {@link OperandCode} for on_value and off_value:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Inputs:
   * * 0: An {@link ANEURALNETWORKS_INT32} tensor, specifying the indices.
   * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the depth.
   * * 2: A tensor, specifying the on_value.
   * * 3: A tensor, specifying the off_value.
   * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the axis to fill. Optional.
   *      (default: -1, a new inner-most axis).
   *
   * Outputs:
   * * 0: The one-hot tensor.
   */
  ANEURALNETWORKS_ONE_HOT_EX = 50023,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_GREATER_EQUAL_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_GREATER_EQUAL instead
   *
   */
  ANEURALNETWORKS_GREATER_EQUAL_EX = 50024,

  /**
   *
   * IMPORTANT NOTICE:
   * ANEURALNETWORKS_LESS_EX operation is DEPRECATED
   * Use ANEURALNETWORKS_LESS instead
   *
   */
  ANEURALNETWORKS_LESS_EX = 50025,

  /**
   * Returns the input tensor's shape as a rank 1 output tensor
   * If the input shape is [D0, D1, ..., D(N-1) ] and rank is N,
   * the output tensor is [D0, D1, ... D(N-1)], shape is [N] and rank is 1.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_INT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: virtually unlimited
   *
   * Inputs:
   * * 0: The input tensor.
   *
   * Outputs:
   * * 0: The rank-1 shape tensor.
   */
  ANEURALNETWORKS_SHAPE_EX = 50026,

  /**
   * Computes element-wise round() on the input tensor.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   *
   * Outputs:
   * * 0: The output tensor, of the same {@link OperandCode} and dimensions as
   *      the input tensor.
   */
  ANEURALNETWORKS_ROUND_EX = 50027,

  /**
   * Reverses specific dimensions of a tensor.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor to reverse.
   * * 1: A 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}. The dimensions to reverse
   *
   * Outputs:
   * * 0: The output tensor, of the same {@link OperandCode} and dimensions as
   *      the input tensor.
   */
  ANEURALNETWORKS_REVERSE_EX = 50028,

  ANEURALNETWORKS_FILL_EX = 50029,

  ANEURALNETWORKS_SELECT_V2_EX = 50030,

  /**
   * Make the output tensor same shape to the input tensor and fill zero for all elements.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32, ANEURALNETWORKS_TENSOR_INT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   *
   * Outputs:
   * * 0: The output tensor, of the same {@link OperandCode} and dimensions as
   *      the input tensor.
   */
  ANEURALNETWORKS_ZEROS_LIKE_EX = 50031,

  /**
   * Computes cosine of x element-wise
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   *
   * Outputs:
   * * 0: The output tensor, of the same {@link OperandCode} and dimensions as
   *      the input tensor.
   */
  ANEURALNETWORKS_COS_EX = 50032,

  /**
   * Creates a sequence of numbers
   * that begins at 'start' and extends by increments of 'delta' up to but not including 'limit'.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32, ANEURALNETWORKS_TENSOR_INT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A 0-D Tensor (scalar). 'start' acts as first entry in the range
   * * 1: A 0-D Tensor (scalar). 'limit' acts as upper limit of sequence
   * * 2: A 0-D Tensor (scalar). 'delta' acts as number that increments 'start'
   *
   * Outputs:
   * * 0: An 1-D output tensor
   */
  ANEURALNETWORKS_RANGE_EX = 50033,

  ANEURALNETWORKS_FUSED_BATCH_NORM_V3_EX = 50034,

  ANEURALNETWORKS_BATCH_MATMUL_EX = 50035,

  /**
   * Copy a tensor setting everything outside a central band in each innermost matrix.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   * * 1: An {@link ANEURALNETWORKS_INT32} scalar. Number of subdiagonals to keep. If negative, keep
   * entire lower triangle.
   * * 2: An {@link ANEURALNETWORKS_INT32} scalar. Number of superdiagonals to keep. If negative,
   * keep entire upper triangle.
   *
   * Outputs:
   * * 0: An output tensor. The extracted banded tensor with the same shape as input.
   */
  ANEURALNETWORKS_MATRIX_BAND_PART_EX = 50036,

  /**
   * Tensor contraction over specified indices and outer product
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0 ~ n-1: The list of n input tensors.
   * * 1: An 1-D tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}. Each element represent
   * equation character.
   *      Always scalar is 1.0 and zeroPoint is 0
   *
   * Outputs:
   * * 0: An output tensor.
   */
  ANEURALNETWORKS_EINSUM_EX = 50037,

  ANEURALNETWORKS_BROADCAST_TO_EX = 50038,

  /** Adds two tensors, element-wise.
   *
   * Takes two input tensors of identical type and compatible dimensions. The output
   * is the sum of both input tensors, optionally modified by an activation function.
   *
   * Two dimensions are compatible when:
   *     1. they are equal, or
   *     2. one of them is 1
   *
   * The size of the output is the maximum size along each dimension of the input operands.
   * It starts with the trailing dimensions, and works its way forward.
   *
   * Example:
   *
   *     input1.dimension = {4, 1, 2}
   *     input2.dimension = {5, 4, 3, 1}
   *     output.dimension = {5, 4, 3, 2}
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   * * 1: A tensor of the same type, and compatible dimensions as input0.
   * * 2: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The sum, a tensor of the same type as input0.
   */
  ANEURALNETWORKS_ADDV2_EX = 50039,

  ANEURALNETWORKS_STATELESS_RANDOM_UNIFORM_EX = 50040,

  /** Splits a tensor value into a list of sub tensors.
   *
   * Supported tensor {@link OperandCode}:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32, ANEURALNETWORKS_TENSOR_INT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor to split.
   * * 1: A tensor containing the sizes of each output tensor along split_dim
   * * 2: The dimension along which to split
   *
   * Outputs:
   * * 0: Tensor objects resulting from splitting value.
   */
  ANEURALNETWORKS_SPLIT_V_EX = 50041

} OperationCodeEx; // extends OperationCode

typedef OperationCodeEx ANeuralNetworksOperationTypeEx;

/**
 * @brief Add an extended operation to a model.
 *
 * @param[in] model The model to be modified.
 * @param[in] type The type of extended operation.
 * @param[in] inputCount The number of entries in the inputs array.
 * @param[in] inputs An array of indexes identifying each operand.
 * @param[in] outputCount The number of entries in the outputs array.
 * @param[in] outputs An array of indexes identifying each operand.
 *
 * @note The operands specified by inputs and outputs must have been
 *       previously added by calls to {@link ANeuralNetworksModel_addOperand}.\n
 *       Attempting to modify a model once {@link ANeuralNetworksModel_finish} has been
 *       called will return an error.\n
 *       See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_addOperationEx(ANeuralNetworksModel *model,
                                        ANeuralNetworksOperationTypeEx type, uint32_t inputCount,
                                        const uint32_t *inputs, uint32_t outputCount,
                                        const uint32_t *outputs);

__END_DECLS

#endif // NN_RUNTIME_NEURAL_NETWORKS_EX_H
