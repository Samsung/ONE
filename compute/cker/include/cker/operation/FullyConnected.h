/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_FULLY_CONNECTED_H__
#define __NNFW_CKER_FULLY_CONNECTED_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include "cker/TensorUtils.h"

namespace nnfw
{
namespace cker
{

class FCTempArena
{
public:
  FCTempArena(void) : prepared(false), input_quantized(), scaling_factors(), accum_scratch()
  {
    // DO NOTHING
  }

  void prepare(const Shape &input_shape, const Shape &weights_shape)
  {
    auto input_size = input_shape.FlatSize();
    input_quantized.resize(input_size);

    assert(weights_shape.DimensionsCount() == 2);
    int batch_size = input_size / weights_shape.Dims(1);
    scaling_factors.resize(batch_size);
    prepared = true;
  }

public:
  bool prepared;
  std::vector<int8_t> input_quantized;
  std::vector<float> scaling_factors;
  std::vector<int32_t> accum_scratch;
};

inline void FullyConnected(const FullyConnectedParams &params, const Shape &input_shape,
                           const float *input_data, const Shape &weights_shape,
                           const float *weights_data, const Shape &, const float *bias_data,
                           const Shape &, float *output_data)
{
  int total_input_size = input_shape.FlatSize();
  int input_size = weights_shape.Dims(1);
  const int batch_size = total_input_size / input_size;
  const int num_units = weights_shape.Dims(0);

  // Output = bias if bias tensor exists.
  if (bias_data)
  {
    VectorBatchVectorAssign(bias_data, num_units, batch_size, output_data);
  }
  else
  {
    ZeroVector(output_data, batch_size * num_units);
  }

  // Compute output += weight * input
  MatrixBatchVectorMultiplyAccumulate(weights_data, num_units, input_size, input_data, batch_size,
                                      output_data, /*result_stride=*/1);

  if (params.activation != FusedActivationFunctionType::kNone)
  {
    // Apply activation function
    ApplyActivationToVector(output_data, batch_size * num_units, params.activation, output_data);
  }
}

inline void FullyConnected(const FullyConnectedParams &params, const Shape &input_shape,
                           const uint8_t *input_data, const Shape &filter_shape,
                           const uint8_t *filter_data, const Shape &bias_shape,
                           const int32_t *bias_data, const Shape &output_shape,
                           uint8_t *output_data)
{
  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(bias_shape);
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  assert(filter_shape.DimensionsCount() >= 2);
  assert(output_shape.DimensionsCount() >= 1);

  assert(output_activation_min <= output_activation_max);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth =
      MatchingDim(filter_shape, filter_dim_count - 2, output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b)
  {
    for (int out_c = 0; out_c < output_depth; ++out_c)
    {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d)
      {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data)
      {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8_t>(acc);
    }
  }
}

inline void FullyConnectedHybrid(const FullyConnectedParams &params, const Shape &input_shape,
                                 const float *input_data, const Shape &filter_shape,
                                 const int8_t *filter_data, const Shape &, const float *bias_data,
                                 const Shape &output_shape, float *output_data,
                                 FCTempArena &temp_arena)
{
  int total_input_size = input_shape.FlatSize();
  const int input_size = filter_shape.Dims(1);
  const int batch_size = total_input_size / input_size;
  const int num_units = filter_shape.Dims(0);

  // Output = bias if bias tensor exists.
  if (bias_data)
  {
    VectorBatchVectorAssign(bias_data, num_units, batch_size, output_data);
  }
  else
  {
    ZeroVector(output_data, batch_size * num_units);
  }

  // Save matrix multiplication computation for all zero input.
  if (IsZeroVector(input_data, total_input_size))
  {
    ApplyActivationToVector(output_data, batch_size * num_units, params.activation, output_data);
    return;
  }

  // Quantize input from float to uint8 + quantization params (scaling factor).
  float unused_min, unused_max;
  float *scaling_factors_ptr = temp_arena.scaling_factors.data();
  int8_t *quant_data = temp_arena.input_quantized.data();

  // Quantize each batch independently.
  for (int b = 0; b < batch_size; ++b)
  {
    const int offset = b * input_size;
    SymmetricQuantizeFloats(input_data + offset, input_size, quant_data + offset, &unused_min,
                            &unused_max, &scaling_factors_ptr[b]);
    // Incorporate scaling of the filter.
    scaling_factors_ptr[b] *= params.weights_scale;
  }

// Compute output += weight * quantized_input
#ifdef USE_RUY_GEMV
  auto output_size = output_shape.FlatSize();
  temp_arena.accum_scratch.resize(output_size);
  int32_t *scratch = temp_arena.accum_scratch.data();
  MatrixBatchVectorMultiplyAccumulate(filter_data, num_units, input_size, quant_data,
                                      scaling_factors_ptr, batch_size, scratch, output_data,
                                      /*result_stride=*/1);
#else
  MatrixBatchVectorMultiplyAccumulate(filter_data, num_units, input_size, quant_data,
                                      scaling_factors_ptr, batch_size, output_data,
                                      /*result_stride=*/1);
  UNUSED_RELEASE(output_shape);
#endif

  // Apply activation function to floats.
  ApplyActivationToVector(output_data, batch_size * num_units, params.activation, output_data);
  return;
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_FULLY_CONNECTED_H__
