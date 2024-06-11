/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_CORE_KERNEL_DATA_H
#define ONERT_MICRO_CORE_KERNEL_DATA_H

#include "OMStatus.h"
#include "reader/OMCircleReader.h"

namespace onert_micro
{
namespace core
{

enum class BroadcastableOpCategory : uint8_t
{
  kNone,
  kNonBroadcast,              // Matching input shapes.
  kFirstInputBroadcastsFast,  // Fivefold nested loops.
  kSecondInputBroadcastsFast, // Fivefold nested loops.
  kGenericBroadcast,          // Fall-back.
  kScalarFirstBroadcast,      // Scalar
  kScalarSecondBroadcast,     // Scalar
};

struct ConcatenationParams
{
  uint32_t num_inputs;
  uint32_t axis;
};

struct SoftmaxParams
{
  float beta;
  int32_t input_multiplier;
  int32_t input_left_shift;
  int diff_min;
  int num_rows;
  int row_size;
};

struct LogSoftmaxParams
{
  int num_rows;
  int row_size;
};

struct L2NormalizationParams
{
  int num_rows;
  int row_size;
  float epsilon = 1e-6;
};

struct Pool2DParams
{
  int32_t stride_w;
  int32_t stride_h;
  int32_t dilation_width_factor;
  int32_t dilation_height_factor;
  int32_t filter_h;
  int32_t filter_w;
  int32_t pad_h;
  int32_t pad_w;
  float activation_min;
  float activation_max;
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
};

struct StridedSliceParams
{
  int8_t start_indices_count;
  int32_t start_indices[5];
  int8_t stop_indices_count;
  int32_t stop_indices[5];
  int8_t strides_count;
  int32_t strides[5];

  int16_t begin_mask;
  int16_t ellipsis_mask;
  int16_t end_mask;
  int16_t new_axis_mask;
  int16_t shrink_axis_mask;
};

struct TransposeParams
{
  int32_t perm_count;
  int32_t perm[5];
};

struct BinaryArithmeticBroadcastParams
{
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  int32_t int32_activation_min;
  int32_t int32_activation_max;
  int64_t int64_activation_min;
  int64_t int64_activation_max;
  BroadcastableOpCategory broadcast_category;
};

struct FloatConv2D
{
  int32_t stride_w;
  int32_t stride_h;
  int32_t dilation_width_factor;
  int32_t dilation_height_factor;
  int32_t depth_multiplier;
  int32_t pad_h;
  int32_t pad_w;
  float activation_min;
  float activation_max;
};

struct FullyConnectedParams
{
  // float activation params.
  float float_activation_min;
  float float_activation_max;

  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
};

struct ComparisonParams
{
  // uint8_t inference params.
  int left_shift;
  int32_t input1_offset;
  int32_t input1_multiplier;
  int input1_shift;
  int32_t input2_offset;
  int32_t input2_multiplier;
  int input2_shift;
  // Shape dependent / common to inference types.
  bool is_broadcast;
};

} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_KERNEL_DATA_H
