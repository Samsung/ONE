/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_EINSUM_HELPER_MATMUL_BCAST_H__
#define __NNFW_CKER_EINSUM_HELPER_MATMUL_BCAST_H__

#include <vector>
#include <memory>
#include <numeric>

#include "BCast.h"
#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

// Simple wrapper over BCast specialized for MatMul.
// Provides utilities for broadcasting across batch dimensions for binary
// MatMul-like operations.

// Fix: Use Shape directly instead of Vec
class MatMulBCast
{
public:
  MatMulBCast(Shape &shape_x, Shape &shape_y)
  {
    if (shape_x.DimensionsCount() < 2 || shape_y.DimensionsCount() < 2)
      return;

    std::vector<int32_t> x;
    std::vector<int32_t> y;

    x.resize(shape_x.DimensionsCount() - 2);
    y.resize(shape_y.DimensionsCount() - 2);

    for (size_t i = 0; i < x.size(); i++)
    {
      x[i] = shape_x.Dims(i);
    }
    for (size_t i = 0; i < y.size(); i++)
    {
      y[i] = shape_y.Dims(i);
    }

    _batch_bcast = std::make_unique<BCast>(std::move(x), std::move(y));
    if (!_batch_bcast->IsValid())
      return;

    const auto &x_reshaped = _batch_bcast->x_reshape();
    const auto &y_reshaped = _batch_bcast->y_reshape();
    auto output_shape = _batch_bcast->output_shape();

    _x_batch_size = std::accumulate(x_reshaped.cbegin(), x_reshaped.cend(), INT32_C(1),
                                    std::multiplies<int32_t>());
    _y_batch_size = std::accumulate(y_reshaped.cbegin(), y_reshaped.cend(), INT32_C(1),
                                    std::multiplies<int32_t>());
    _output_shape.ReplaceWith(output_shape.size(), output_shape.data());
    _output_batch_size = _output_shape.FlatSize();
  }

  bool IsValid() const { return (_batch_bcast != nullptr) && _batch_bcast->IsValid(); }
  int32_t x_batch_size() const { return _x_batch_size; }
  int32_t y_batch_size() const { return _y_batch_size; }
  int32_t output_batch_size() const { return _output_batch_size; }
  const Shape &output_batch_shape() const { return _output_shape; }

private:
  std::unique_ptr<BCast> _batch_bcast;

  int32_t _x_batch_size;
  int32_t _y_batch_size;
  Shape _output_shape;
  int32_t _output_batch_size;
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EINSUM_HELPER_MATMUL_BCAST_H__
