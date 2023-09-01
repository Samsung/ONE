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

// (output_shape, output_data, arg_max_index) = MaxPool2D(input_shape, input_data)
inline void MaxPool2D(const PoolParams &params, const Shape &input_shape, const float *input_data,
                      const Shape &output_shape, float *output_data, int *arg_max_index)
{
  assert(input_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);
  assert(input_shape.Dims(3) == output_shape.Dims(3)); // MaxPool2D doesn't change depth

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  // Convert to (D, B*H*W) matrix, Since MaxPool2D input, output element resizes in same depth.
  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  auto matched_index_mat = MapAsMatrixWithLastDimAsRows(arg_max_index, output_shape);

  // Prefill the output to minimum representable float value
  out_mat.setConstant(std::numeric_limits<float>::lowest());

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
            const int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            const int in_offset = NodeOffset(b, h, w, input_height, input_width);

            const auto out_vector = out_mat.col(out_offset);
            const auto in_vector = in_mat.col(in_offset);

            // update input_index
            matched_index_mat.col(out_offset) =
              (out_vector.array() < in_vector.array())
                .select(in_offset, matched_index_mat.col(out_offset));

            // update output matrix
            out_mat.col(out_offset) = out_vector.cwiseMax(in_vector);
          }
        }
      }
    }
  }
}

// (deriv_input_shape, deriv_input_data) = MaxPool2DGrad(output_shape, deriv_ouput,
// output_arg_max_index)
inline void MaxPool2DGrad(const Shape &deriv_output_shape, const float *deriv_output_data,
                          const int *arg_max_index, const Shape &deriv_input_shape,
                          float *deriv_input_data)
{
  assert(deriv_input_shape.DimensionsCount() == 4);
  assert(deriv_output_shape.DimensionsCount() == 4);

  // initialize deriv_input_data
  memset(deriv_input_data, 0, sizeof(float) * deriv_input_shape.FlatSize());

  const int depth = MatchingDim(deriv_input_shape, 3, deriv_output_shape, 3);
  const auto deriv_out_mat = MapAsMatrixWithLastDimAsRows(deriv_output_data, deriv_output_shape);
  auto arg_max_index_mat = MapAsMatrixWithLastDimAsRows(arg_max_index, deriv_output_shape);
  auto deriv_in_mat = MapAsMatrixWithLastDimAsRows(deriv_input_data, deriv_input_shape);

  for (int col_index = 0; col_index < deriv_out_mat.cols(); col_index++)
  {
    auto indices = arg_max_index_mat.col(col_index);
    for (int d = 0; d < depth; d++)
    {
      deriv_in_mat(d, indices(d)) += deriv_out_mat(d, col_index);
    }
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_MAXPOOL_H__
