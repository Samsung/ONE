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

#ifndef __NNFW_CKER_CONV_H__
#define __NNFW_CKER_CONV_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"
#include "cker/operation/reference/Conv.h"
#include "cker/operation/optimized/Conv.h"
#include <iostream>
#include <vector>

namespace nnfw
{
namespace cker
{

namespace
{
// Naive implementation of transpose for floats. Could be optimized to be more
// cache friendly, but for now it's a one-time cost on first run, and we would
// prefer to remove the need to do this at all eventually.
inline void TransposeFloatTensor(const float *input_data, const nnfw::cker::Shape &output_shape,
                                 float *output_data)
{
  const int rows = output_shape.Dims(1);
  const int cols = output_shape.Dims(0);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      const float in_value = input_data[i * cols + j];
      output_data[j * rows + i] = in_value;
    }
  }
}
} // namespace

class Conv
{
public:
  Conv() : _modified_filter_data(), _im2col_shape(4), _need_im2col(false), _prepared(false) {}

  void prepareF32(const Shape &filter_shape, const float *filter_data, PaddingType padding_type,
                  bool &is_replaced_weights, uint32_t dilationWidthFactor,
                  uint32_t dilationHeightFactor)
  {
    if (!_prepared)
    {
      if (usableMultiThreaded(padding_type, dilationWidthFactor, dilationHeightFactor))
      {
        transposeFilter(filter_shape, filter_data, is_replaced_weights);
      }
      _prepared = true;
    }
  }

  void prepareQ8uPerTensor(const Shape &input_shape, const Shape &kernel_shape,
                           const Shape &output_shape, uint32_t stride_width, uint32_t stride_height,
                           uint32_t dilation_width_factor, uint32_t dilation_height_factor)
  {
    if (!_prepared)
    {
      IsRequiredIm2col(input_shape, kernel_shape, output_shape, stride_width, stride_height,
                       dilation_width_factor, dilation_height_factor);
      _prepared = true;
    }
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const float *input_data,
                  const Shape &filter_shape, const float *filter_data, const Shape &bias_shape,
                  const float *bias_data, const Shape &output_shape, float *output_data)
  {
    if (usableMultiThreaded(params.padding_type, params.dilation_width_factor,
                            params.dilation_height_factor))
    {
      bool transposed_in_execution = false;
      if (!_prepared)
      {
        // This means that filter is not constant
        // TODO Apply optimized kernel if multithreaded kernel is slower than optimized kernel by
        // transposing filter data
        transposeFilter(filter_shape, filter_data, transposed_in_execution);
      }
      multithreaded::Conv(params, input_shape, input_data, filter_shape, &_modified_filter_data[0],
                          bias_shape, bias_data, output_shape, output_data);
    }
    else
    {
      // TODO Support optimized kernel
      reference::Conv(params, input_shape, input_data, filter_shape, filter_data, bias_shape,
                      bias_data, output_shape, output_data);
    }
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const uint8_t *input_data,
                  const Shape &filter_shape, const uint8_t *filter_data, const Shape &bias_shape,
                  const int32_t *bias_data, const Shape &output_shape, uint8_t *output_data)
  {
    if (!_prepared)
    {
      // This means that input or output are dynamic or filter is not constant
      IsRequiredIm2col(input_shape, filter_shape, output_shape, params.stride_width,
                       params.stride_height, params.dilation_width_factor,
                       params.dilation_height_factor);
    }

    int im2col_size = _need_im2col ? _im2col_shape.FlatSize() : 1;

    // Use heap if size is larger than 8MB
    if (im2col_size > 8 * 1024 * 1024)
    {
      std::unique_ptr<uint8_t[]> im2col_data = std::make_unique<uint8_t[]>(im2col_size);
      optimized::Conv(params, input_shape, input_data, filter_shape, filter_data, bias_shape,
                      bias_data, output_shape, output_data, _im2col_shape, im2col_data.get());
    }
    else
    {
      uint8_t im2col_data[im2col_size];
      optimized::Conv(params, input_shape, input_data, filter_shape, filter_data, bias_shape,
                      bias_data, output_shape, output_data, _im2col_shape, im2col_data);
    }
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const uint8_t *input_data,
                  const Shape &filter_shape, const uint8_t *filter_data,
                  const int32_t *filter_zero_point, const Shape &bias_shape,
                  const int32_t *bias_data, const Shape &output_shape, uint8_t *output_data)
  {
    reference::Conv<uint8_t, true>(params, _per_channel_output_multiplier.data(),
                                   _per_channel_output_shift.data(), input_shape, input_data,
                                   filter_shape, filter_data, filter_zero_point, bias_shape,
                                   bias_data, output_shape, output_data);
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const int8_t *input_data,
                  const Shape &filter_shape, const int8_t *filter_data, const Shape &bias_shape,
                  const int32_t *bias_data, const Shape &output_shape, int8_t *output_data)
  {
    reference::Conv<int8_t, false>(params, _per_channel_output_multiplier.data(),
                                   _per_channel_output_shift.data(), input_shape, input_data,
                                   filter_shape, filter_data, nullptr /* filter_zero_point */,
                                   bias_shape, bias_data, output_shape, output_data);
  }
  std::vector<int32_t> &per_channel_output_multiplier() { return _per_channel_output_multiplier; }
  std::vector<int> &per_channel_output_shift() { return _per_channel_output_shift; }

private:
  bool usableMultiThreaded(PaddingType padding_type, uint32_t dilation_width_factor,
                           int32_t dilation_height_factor)
  {
    return padding_type != PaddingType::kNone && std::thread::hardware_concurrency() > 1 &&
           dilation_width_factor == 1 && dilation_height_factor == 1;
  }

  void transposeFilter(const Shape &filter_shape, const float *filter_data,
                       bool &is_replaced_weights)
  {
    const auto output_depth = filter_shape.Dims(0);
    const Shape hwcn_filter_shape{filter_shape.FlatSize() / output_depth, output_depth};
    _modified_filter_data.resize(hwcn_filter_shape.FlatSize());
    TransposeFloatTensor(filter_data, hwcn_filter_shape, &_modified_filter_data[0]);
    is_replaced_weights = true;
  }

  void IsRequiredIm2col(const Shape &input_shape, const Shape &kernel_shape,
                        const Shape &output_shape, uint32_t stride_width, uint32_t stride_height,
                        uint32_t dilation_width_factor, uint32_t dilation_height_factor)
  {
    const bool need_dilated_im2col = dilation_width_factor != 1 || dilation_height_factor != 1;
    const bool need_non_dilated_im2col = stride_width != 1 || stride_height != 1 ||
                                         kernel_shape.Dims(1) != 1 || kernel_shape.Dims(2) != 1;

    _need_im2col = need_dilated_im2col || need_non_dilated_im2col;

    if (_need_im2col)
    {
      _im2col_shape.SetDim(0, output_shape.Dims(0));
      _im2col_shape.SetDim(1, output_shape.Dims(1));
      _im2col_shape.SetDim(2, output_shape.Dims(2));
      _im2col_shape.SetDim(3, input_shape.Dims(3) * kernel_shape.Dims(1) * kernel_shape.Dims(2));
    }
  }

private:
  std::vector<float> _modified_filter_data;
  Shape _im2col_shape;
  bool _need_im2col;
  bool _prepared;
  // Per channel output multiplier and shift.
  std::vector<int32_t> _per_channel_output_multiplier;
  std::vector<int> _per_channel_output_shift;
};

struct ConvHybridTempArena
{
  ConvHybridTempArena(int input_size, int batch_size)
  {
    input_quantized.resize(input_size);
    // TODO: Optimize the case of batch_size = 1
    input_scaling_factors.resize(batch_size);
    input_offsets.resize(batch_size);
  }
  std::vector<int8_t> input_quantized;
  std::vector<float> input_scaling_factors;
  std::vector<int32_t> input_offsets;
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_CONCATENATION_H_
