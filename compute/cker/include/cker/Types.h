/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_TYPES_H__
#define __NNFW_CKER_TYPES_H__

#include <cstdint>

namespace nnfw
{
namespace cker
{

enum class FusedActivationFunctionType
{
  kNone = 0,
  kRelu6 = 1,
  kRelu1 = 2,
  kRelu = 3,
};
enum class PaddingType
{
  kNone = 0,
  kSame = 1,
  kValid = 2,
};

enum class BinaryArithmeticOpType
{
  ADD = 0,
  SUB = 1,
  MUL = 2,
  DIV = 3,
};

enum class ComparisonOpType
{
  Equal,
  NotEqual,
  Greater,
  GreaterEqual,
  Less,
  LessEqual
};

struct PaddingValues
{
  int16_t width;
  int16_t height;
};

struct PoolParams
{
  FusedActivationFunctionType activation;
  PaddingType padding_type;
  PaddingValues padding_values;
  int stride_height;
  int stride_width;
  int filter_height;
  int filter_width;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct SoftmaxParams
{
  // beta is not really used (not a Tensorflow parameter) and not implemented
  // for LogSoftmax.
  double beta;
  // uint8 inference params.  Used even when beta defaults to 1.0.
  int32_t input_multiplier;
  int32_t input_left_shift;
  // Reverse scaling is only used by LogSoftmax.
  int32_t reverse_scaling_divisor;
  int32_t reverse_scaling_right_shift;
  int diff_min;
};

struct PackParams
{
  int8_t axis;
  // zeropoint and scale were only used to implement PackWithScaling in the legacy code of
  // tensorflow
  // const int32_t* input_zeropoint;
  // const float* input_scale;
  uint16_t inputs_count;
  // int32_t output_zeropoint;
  // float output_scale;
};

struct UnpackParams
{
  uint16_t num_split;
  int16_t axis;
};

struct ConvParams
{
  PaddingType padding_type;
  PaddingValues padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  bool is_replaced_weights{false};
};

struct ComparisonParams
{
  ComparisonOpType type;
  bool is_broadcast;
};

struct BinaryArithmeticOpParam
{
  BinaryArithmeticOpType type;
  // Shape dependent / common to data / op types.
  // BroadcastableOpCategory broadcast_category;
  // uint8 inference params.
  int32_t input1_offset;
  int32_t input2_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int32_t output_shift;
  // Add / Sub, not Mul, uint8 inference params.
  int32_t left_shift;
  int32_t input1_multiplier;
  int32_t input1_shift;
  int32_t input2_multiplier;
  int32_t input2_shift;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;

  // Processed output dimensions.
  // Let input "a" be the one that broadcasts in the faster-changing dimension.
  // Then, after coalescing, for shapes {a0, a1, a2, a3, a4} and
  // {b0, b1, b2, b3, b4},
  // broadcast_shape[4] = b0 = a0.
  // broadcast_shape[3] = b1; a1 = 1.
  // broadcast_shape[2] = b2 = a2.
  // broadcast_shape[1] = a3; b3 = 1.
  // broadcast_shape[0] = b4 = a4.
  // int broadcast_shape[5];
};

struct TransposeParams
{
  int8_t perm_count;
  int32_t perm[4];
};

struct ConcatenationParams
{
  int8_t axis;
  const int32_t *input_zeropoint;
  const float *input_scale;
  uint16_t inputs_count;
  int32_t output_zeropoint;
  float output_scale;
};

struct DepthwiseConvParams
{
  PaddingType padding_type;
  PaddingValues padding_values;
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  int16_t depth_multiplier;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct FullyConnectedParams
{
  FusedActivationFunctionType activation;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  float weights_scale;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  // FullyConnectedWeightsFormat weights_format;
};

struct GatherParams
{
  int32_t axis;
};

struct InstanceNormParams
{
  float epsilon;
  float float_activation_min;
  float float_activation_max;
};

struct TransposeConvParams
{
  PaddingType padding_type;
  PaddingValues padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct SliceParams
{
  int8_t begin_count;
  int32_t begin[4];
  int8_t size_count;
  int32_t size[4];
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TYPES_H__
