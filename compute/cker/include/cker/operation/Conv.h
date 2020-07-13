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
  Conv()
      : _modified_filter_data(), _im2col_data(), _im2col_shape(4), _need_im2col(false),
        _prepared(false)
  {
  }

  void prepare(const Shape &filter_shape, const float *filter_data, PaddingType padding_type,
               bool &is_replaced_weights)
  {
    if (!_prepared)
    {
      if (usableMultiThreaded(padding_type))
      {
        transposeFilter(filter_shape, filter_data, is_replaced_weights);
      }
      _prepared = true;
    }
  }

  void prepareQuant(const Shape &input_shape, const Shape &kernel_shape, const Shape &output_shape,
                    uint32_t stride_width, uint32_t stride_height)
  {
    if (!_prepared)
    {
      IsRequiredIm2col(input_shape, kernel_shape, output_shape, stride_width, stride_height);
      _prepared = true;
    }
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const float *input_data,
                  const Shape &filter_shape, const float *filter_data, const Shape &bias_shape,
                  const float *bias_data, const Shape &output_shape, float *output_data)
  {
    if (usableMultiThreaded(params.padding_type))
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
                       params.stride_height);
    }

    uint8_t *im2col_raw_data = _im2col_data.data();
    optimized::Conv(params, input_shape, input_data, filter_shape, filter_data, bias_shape,
                    bias_data, output_shape, output_data, _im2col_shape, im2col_raw_data);
  }

private:
  bool usableMultiThreaded(PaddingType padding_type)
  {
    return padding_type != PaddingType::kNone && std::thread::hardware_concurrency() > 1;
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
                        const Shape &output_shape, uint32_t stride_width, uint32_t stride_height)
  {
    _need_im2col = stride_width != 1 || stride_height != 1 || kernel_shape.Dims(1) != 1 ||
                   kernel_shape.Dims(2) != 1;
    if (_need_im2col)
    {
      _im2col_shape.SetDim(0, output_shape.Dims(0));
      _im2col_shape.SetDim(1, output_shape.Dims(1));
      _im2col_shape.SetDim(2, output_shape.Dims(2));
      _im2col_shape.SetDim(3, input_shape.Dims(3) * kernel_shape.Dims(1) * kernel_shape.Dims(2));
      _im2col_data.resize(_im2col_shape.FlatSize());
    }
  }

private:
  std::vector<float> _modified_filter_data;
  std::vector<uint8_t> _im2col_data;
  Shape _im2col_shape;
  bool _need_im2col;
  bool _prepared;
};
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_CONCATENATION_H_
