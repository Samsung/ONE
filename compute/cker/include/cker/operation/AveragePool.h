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

#ifndef __NNFW_CKER_AVERAGE_POOL_H__
#define __NNFW_CKER_AVERAGE_POOL_H__

#include "cker/eigen/Utils.h"
#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include <Eigen/Core>

namespace nnfw
{
namespace cker
{

// TODO Change to apply neon for this function if it is faster
inline void AveragePool(const PoolParams &params, const Shape &input_shape, const float *input_data,
                        const Shape &output_shape, float *output_data)
{
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  // TODO(benoitjacob) make this a proper reference impl without Eigen!
  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // TODO(benoitjacob) get rid of the dynamic memory allocation here!
  Eigen::VectorXf out_count(out_mat.cols());
  out_count.setZero();
  // Prefill the output to 0.
  out_mat.setZero();
  for (int b = 0; b < batches; ++b)
  {
    for (int h = 0; h < input_height; ++h)
    {
      for (int w = 0; w < input_width; ++w)
      {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + params.padding_values.height;
        int wpad = w + params.padding_values.width;
        int h_start =
            (hpad < params.filter_height) ? 0 : (hpad - params.filter_height) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start =
            (wpad < params.filter_width) ? 0 : (wpad - params.filter_width) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);
        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph)
        {
          for (int pw = w_start; pw < w_end; ++pw)
          {
            int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            out_mat.col(out_offset) += in_mat.col(NodeOffset(b, h, w, input_height, input_width));
            out_count(out_offset)++;
          }
        }
      }
    }
  }
  // Divide the output by the actual number of elements being averaged over
  assert(out_count.minCoeff() > 0);
  out_mat.array().rowwise() /= out_count.transpose().array();

  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i], params.float_activation_min,
                                                  params.float_activation_max);
  }
}

inline void AveragePool(const PoolParams &params, const Shape &input_shape,
                        const uint8_t *input_data, const Shape &output_shape, uint8_t *output_data)
{
  assert(params.quantized_activation_min <= params.quantized_activation_max);
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
        const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
        // Compute the boundaries of the filter region clamped so as to
        // ensure that the filter window fits in the input array.
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
        int filter_count = (filter_y_end - filter_y_start) * (filter_x_end - filter_x_start);
        if (filter_count <= 0)
        {
          continue;
        }
        for (int channel = 0; channel < depth; ++channel)
        {
          int32_t acc = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
          {
            for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x)
            {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              acc += input_data[Offset(input_shape, batch, in_y, in_x, channel)];
            }
          }
          acc = (acc + filter_count / 2) / filter_count;
          acc = std::max(acc, params.quantized_activation_min);
          acc = std::min(acc, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<uint8_t>(acc);
        }
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_AVERAGE_POOL_H__
