/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "NodeExecution.h"

#include "NodeDataImpl.h"
#include "NodeDomain.h"
#include "Validation.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <limits>
#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace
{

/**
 * @brief Compute 1D output size based on given 1D arguments.
 *
 * @param whole_pad Sum of front and back pad
 */
inline uint32_t compute_out_size(uint32_t image_size, uint32_t whole_pad, uint32_t filter_size,
                                 uint32_t stride)
{
  assert((image_size + whole_pad - filter_size) % stride == 0);
  return (image_size + whole_pad - filter_size) / stride + 1;
}

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

template <typename T>
nncc::core::ADT::tensor::Buffer<T> maxPool2D(const loco::MaxPool2D *maxpool2d,
                                             const Buffer<T> *ifm_buf)
{
  auto ifm_shape = ifm_buf->shape();

  const uint32_t batches = ifm_shape.dim(0);
  const uint32_t depth = ifm_shape.dim(3);

  const uint32_t ifm_height = ifm_shape.dim(1);
  const uint32_t ifm_width = ifm_shape.dim(2);

  const uint32_t window_height = maxpool2d->window()->vertical();
  const uint32_t window_width = maxpool2d->window()->horizontal();

  const uint32_t stride_height = maxpool2d->stride()->vertical();
  const uint32_t stride_width = maxpool2d->stride()->horizontal();

  const uint32_t pad_top = maxpool2d->pad()->top();
  const uint32_t pad_bottom = maxpool2d->pad()->bottom();

  const uint32_t pad_left = maxpool2d->pad()->left();
  const uint32_t pad_right = maxpool2d->pad()->right();

  const uint32_t output_height =
      compute_out_size(ifm_height, pad_top + pad_bottom, window_height, stride_height);
  const uint32_t output_width =
      compute_out_size(ifm_width, pad_left + pad_right, window_width, stride_width);

  // prepare output buffer
  Shape output_shape{batches, output_height, output_width, depth};
  auto output_buf = make_buffer<T, LexicalLayout>(output_shape);

  for (uint32_t batch = 0; batch < batches; ++batch)
  {
    for (uint32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (uint32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (uint32_t channel = 0; channel < depth; ++channel)
        {
          const int in_x_origin = (out_x * stride_width) - pad_left;
          const int in_y_origin = (out_y * stride_height) - pad_top;

          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const uint32_t filter_x_start = std::max(0, -in_x_origin);
          const uint32_t filter_x_end = std::min(window_width, ifm_width - in_x_origin);

          const uint32_t filter_y_start = std::max(0, -in_y_origin);
          const uint32_t filter_y_end = std::min(window_height, ifm_height - in_y_origin);

          T max = std::numeric_limits<T>::lowest();

          for (uint32_t filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
          {
            for (uint32_t filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x)
            {
              const uint32_t in_x = in_x_origin + filter_x;
              const uint32_t in_y = in_y_origin + filter_y;
              max = std::max(max, ifm_buf->at(Index({batch, in_y, in_x, channel})));
            }
          }

          output_buf.at(Index({batch, out_y, out_x, channel})) = max;
        }
      }
    }
  }

  return output_buf;
}

} // namespace

namespace
{

using namespace locomotiv;

void execute_node(loco::MaxPool2D *maxpool2d)
{
  auto ifm_data = annot_data(maxpool2d->ifm());

  validate(ifm_data, "Can't find input data of MaxPool2D");
  validate(ifm_data->shape()->rank() == 4, "IFM rank should be 4");
  validate(annot_domain(maxpool2d->ifm()) == loco::Domain::Feature,
           "ifm of MaxPool2D is not Feature");

  std::unique_ptr<NodeData> maxpool2d_data = nullptr;

  switch (ifm_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto ifm_buf = ifm_data->as_f32_bufptr();

      auto maxpool2d_buf = maxPool2D<float>(maxpool2d, ifm_buf);

      maxpool2d_data = make_data(maxpool2d_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(maxpool2d_data != nullptr);

  annot_data(maxpool2d, std::move(maxpool2d_data));
  annot_domain(maxpool2d, loco::Domain::Feature);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::MaxPool2D *maxpool2d) { execute_node(maxpool2d); }

} // namespace locomotiv
