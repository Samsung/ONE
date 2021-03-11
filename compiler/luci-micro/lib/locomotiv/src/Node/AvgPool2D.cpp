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
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <cassert>
#include <stdexcept>

namespace
{

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

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

template <typename T>
nncc::core::ADT::tensor::Buffer<T> avgPool2D(const loco::AvgPool2D *avgpool2d,
                                             const Buffer<T> *ifm_buf)
{
  assert(avgpool2d->convention() == loco::AvgPool2D::Convention::Valid ||
         avgpool2d->convention() == loco::AvgPool2D::Convention::Full);

  auto ifm_shape = ifm_buf->shape();

  const uint32_t batches = ifm_shape.dim(0);
  const uint32_t depth = ifm_shape.dim(3);

  const uint32_t ifm_height = ifm_shape.dim(1);
  const uint32_t ifm_width = ifm_shape.dim(2);

  const uint32_t window_height = avgpool2d->window()->vertical();
  const uint32_t window_width = avgpool2d->window()->horizontal();

  const uint32_t stride_height = avgpool2d->stride()->vertical();
  const uint32_t stride_width = avgpool2d->stride()->horizontal();

  const uint32_t pad_top = avgpool2d->pad()->top();
  const uint32_t pad_bottom = avgpool2d->pad()->bottom();

  const uint32_t pad_left = avgpool2d->pad()->left();
  const uint32_t pad_right = avgpool2d->pad()->right();

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

          uint32_t f_x0, f_x1, f_y0, f_y1;
          if (avgpool2d->convention() == loco::AvgPool2D::Convention::Valid)
          {
            f_x0 = std::max(0, -in_x_origin);
            f_x1 = std::min(window_width, ifm_width - in_x_origin);
            f_y0 = std::max(0, -in_y_origin);
            f_y1 = std::min(window_height, ifm_height - in_y_origin);
          }
          else
          {
            throw std::runtime_error("TODO support AvgPool2D::Convention::Full");
          }
          const uint32_t filter_x_start = f_x0;
          const uint32_t filter_x_end = f_x1;

          const uint32_t filter_y_start = f_y0;
          const uint32_t filter_y_end = f_y1;

          T total = 0;
          uint32_t filter_ele_count = 0;

          for (uint32_t filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
          {
            for (uint32_t filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x)
            {
              const uint32_t in_x = in_x_origin + filter_x;
              const uint32_t in_y = in_y_origin + filter_y;
              total += ifm_buf->at(Index({batch, in_y, in_x, channel}));
              filter_ele_count++;
            }
          }

          if (filter_ele_count <= 0)
            throw std::runtime_error("The number of filter element must be greater than zero.");
          output_buf.at(Index({batch, out_y, out_x, channel})) = total / filter_ele_count;
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

void exectute_node(loco::AvgPool2D *avgpool2d)
{
  auto ifm_data = annot_data(avgpool2d->ifm());

  validate(ifm_data, "Can't find input data of AvgPool2D");
  validate(ifm_data->shape()->rank() == 4, "IFM rank should be 4");
  validate(annot_domain(avgpool2d->ifm()) == loco::Domain::Feature,
           "ifm of AvgPool2D is not Feature");

  std::unique_ptr<NodeData> avgpool2d_data = nullptr;

  switch (ifm_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto ifm_buf = ifm_data->as_f32_bufptr();

      auto avgpool2d_buf = avgPool2D<float>(avgpool2d, ifm_buf);

      avgpool2d_data = make_data(avgpool2d_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(avgpool2d_data != nullptr);

  annot_data(avgpool2d, std::move(avgpool2d_data));
  annot_domain(avgpool2d, loco::Domain::Feature);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::AvgPool2D *avgpool2d) { exectute_node(avgpool2d); }

} // namespace locomotiv
