/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_AVERAGEPOOL_H__
#define __NNFW_CKER_TRAIN_OPERATION_AVERAGEPOOL_H__

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

inline void AveragePool2DGrad(const PoolParams &params, const Shape &incoming_shape,
                              const float *incoming_data, const Shape &grad_shape, float *grad_data)
{
  assert(grad_shape.DimensionsCount() == 4);
  assert(incoming_shape.DimensionsCount() == 4);

  const int batches = MatchingDim(incoming_shape, 0, grad_shape, 0);
  const int grad_height = grad_shape.Dims(1);
  const int grad_width = grad_shape.Dims(2);
  const int incoming_height = incoming_shape.Dims(1);
  const int incoming_width = incoming_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  // initialize grad_data
  std::fill(grad_data, grad_data + grad_shape.FlatSize(), 0.0);

  const auto incoming_mat = MapAsMatrixWithLastDimAsRows(incoming_data, incoming_shape);
  auto grad_mat = MapAsMatrixWithLastDimAsRows(grad_data, grad_shape);

  for (int b = 0; b < batches; ++b)
  {
    for (int h = 0; h < incoming_height; ++h)
    {
      for (int w = 0; w < incoming_width; ++w)
      {
        // (h_start, h_end) * (w_start, w_end) is input range
        // that output is projected from.
        int h_start = h * stride_height - params.padding_values.height;
        int h_end = std::min(h_start + params.filter_height, grad_height);
        h_start = h_start < 0 ? 0 : h_start;

        int w_start = w * stride_width - params.padding_values.width;
        int w_end = std::min(w_start + params.filter_width, grad_width);
        w_start = w_start < 0 ? 0 : w_start;

        int count = (h_end - h_start) * (w_end - w_start);

        if (h_end <= 0 || w_end <= 0 || count <= 0 || h_start >= grad_height ||
            w_start >= grad_width)
          continue;

        int incoming_offset = NodeOffset(b, h, w, incoming_height, incoming_width);
        for (int ph = h_start; ph < h_end; ++ph)
        {
          for (int pw = w_start; pw < w_end; ++pw)
          {
            int grad_offset = NodeOffset(b, ph, pw, grad_height, grad_width);
            grad_mat.col(grad_offset) += incoming_mat.col(incoming_offset) / count;
          }
        }
      }
    }
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_AVERAGEPOOL_H__
