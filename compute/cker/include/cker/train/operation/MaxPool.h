/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_MAXPOOL_H__
#define __NNFW_CKER_TRAIN_OPERATION_MAXPOOL_H__

#include "cker/Shape.h"
#include "cker/Utils.h"
#include "cker/eigen/Utils.h"

#include <Eigen/Core>

namespace nnfw
{
namespace cker
{
namespace train
{

// Most of the logic except 'arg_max_index' related is copy-paste from
// https://github.com/Samsung/ONE/blob/a380292/compute/cker/include/cker/operation/MaxPool.h#L42-L88
// 'arg_max_index' is to record max-arguments' index to apply gradient later.
inline void MaxPool2D(const PoolParams &params, const Shape &input_shape, const float *input_data,
                      const Shape &output_shape, float *output_data, int *arg_max_index)
{
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  assert(input_shape.Dims(0) == output_shape.Dims(0)); // MaxPool2D doesn't change batch
  assert(input_shape.Dims(3) == output_shape.Dims(3)); // MaxPool2D doesn't change depth

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int filter_height = params.filter_height;
  const int filter_width = params.filter_width;
  const int pad_height = params.padding_values.height;
  const int pad_width = params.padding_values.width;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  auto arg_max_index_mat = MapAsMatrixWithLastDimAsRows(arg_max_index, output_shape);

  // initialize output area
  std::fill(output_data, output_data + output_shape.FlatSize(), 0.0);
  std::fill(arg_max_index, arg_max_index + output_shape.FlatSize(), -1);

  // initialize projected area with lowest float
  const int h_start =
    (pad_height < filter_height) ? 0 : (pad_height - filter_height) / stride_height + 1;
  const int h_end = std::min((input_height + pad_height - 1) / stride_height + 1, output_height);

  const int w_start =
    (pad_width < filter_width) ? 0 : (pad_width - filter_width) / stride_width + 1;
  const int w_end = std::min((input_width + pad_width - 1) / stride_width + 1, output_width);

  for (int b = 0; b < batches; ++b)
  {
    for (int h_idx = h_start; h_idx < h_end; h_idx++)
    {
      for (int w_idx = w_start; w_idx < w_end; w_idx++)
      {
        const int offset = NodeOffset(b, h_idx, w_idx, output_height, output_width);
        out_mat.col(offset).setConstant(std::numeric_limits<float>::lowest());
      }
    }
  }

  for (int b = 0; b < batches; ++b)
  {
    for (int h = 0; h < input_height; ++h)
    {
      for (int w = 0; w < input_width; ++w)
      {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + pad_height;
        int wpad = w + pad_width;

        int h_start = (hpad < filter_height) ? 0 : (hpad - filter_height) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);

        int w_start = (wpad < filter_width) ? 0 : (wpad - filter_width) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);

        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph)
        {
          for (int pw = w_start; pw < w_end; ++pw)
          {
            const int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            const int in_offset = NodeOffset(b, h, w, input_height, input_width);

            const auto out_vector = out_mat.col(out_offset);
            const auto in_vector = in_mat.col(in_offset);

            // update arg_max_index_mat
            arg_max_index_mat.col(out_offset) =
              (out_vector.array() < in_vector.array())
                .select(in_offset, arg_max_index_mat.col(out_offset));

            // update out_mat
            out_mat.col(out_offset) = out_vector.cwiseMax(in_vector);
          }
        }
      }
    }
  }

  out_mat.cwiseMin(params.float_activation_min).cwiseMax(params.float_activation_max);
}

inline void MaxPool2DGrad(const Shape &incoming_shape, const float *incoming_data,
                          const int *arg_max_index, const Shape &grad_shape, float *grad_data)
{
  assert(grad_shape.DimensionsCount() == 4);
  assert(incoming_shape.DimensionsCount() == 4);

  // initialize grad_data
  std::fill(grad_data, grad_data + grad_shape.FlatSize(), 0.0);

  const int depth = MatchingDim(grad_shape, 3, incoming_shape, 3);
  const auto incoming_mat = MapAsMatrixWithLastDimAsRows(incoming_data, incoming_shape);
  auto arg_max_index_mat = MapAsMatrixWithLastDimAsRows(arg_max_index, incoming_shape);
  auto grad_mat = MapAsMatrixWithLastDimAsRows(grad_data, grad_shape);

  for (int col_index = 0; col_index < incoming_mat.cols(); col_index++)
  {
    auto arg_indices = arg_max_index_mat.col(col_index);
    for (int d = 0; d < depth; d++)
    {
      // output value is from padding, so nothing to propagate
      if (arg_indices(d) == -1)
        continue;

      grad_mat(d, arg_indices(d)) += incoming_mat(d, col_index);
    }
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_MAXPOOL_H__
