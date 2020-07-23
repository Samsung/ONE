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

#include <cassert>
#include <stdexcept>

namespace
{
// image size includes padding.
inline uint32_t compute_out_size(uint32_t image_size, uint32_t filter_size, uint32_t stride)
{
  assert((image_size + stride - filter_size) % stride == 0);
  return (image_size + stride - filter_size) / stride;
}

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

/**
 * @brief Calculates Conv2D
 * @note  Both input_buf and filter_buf have NHWC format
 */
template <typename RET_T, typename IFM_T, typename FIL_T>
Buffer<RET_T> calc_conv2D(const loco::Conv2D *conv2d, const Buffer<IFM_T> *input_buf,
                          const Buffer<FIL_T> *filter_buf)
{
  auto input_shape = input_buf->shape();
  auto filter_shape = filter_buf->shape();

  locomotiv::validate(input_shape.rank() == 4, "ifm rank must be 4");
  locomotiv::validate(filter_shape.rank() == 4, "filter rank must be 4");
  locomotiv::validate(input_shape.dim(3) == filter_shape.dim(3),
                      "channel value mismatch"); // should have same channel values

  const uint32_t input_height = input_shape.dim(1);
  const uint32_t input_width = input_shape.dim(2);

  const uint32_t filter_height = filter_shape.dim(1);
  const uint32_t filter_width = filter_shape.dim(2);

  const uint32_t stride_width = conv2d->stride()->horizontal();
  const uint32_t stride_height = conv2d->stride()->vertical();

  // TODO Enable dilations. Let's set these to 1 for now.
  const uint32_t dilation_width_factor = 1;
  const uint32_t dilation_height_factor = 1;

  const uint32_t pad_top = conv2d->pad()->top();
  const uint32_t pad_bottom = conv2d->pad()->bottom();

  const uint32_t pad_left = conv2d->pad()->left();
  const uint32_t pad_right = conv2d->pad()->right();

  const uint32_t output_height =
      compute_out_size(input_height + pad_top + pad_bottom, filter_height, stride_height);
  const uint32_t output_width =
      compute_out_size(input_width + pad_left + pad_right, filter_width, stride_width);

  const uint32_t batches = input_shape.dim(0);
  const uint32_t input_depth = input_shape.dim(3);
  const uint32_t output_depth = filter_shape.dim(0);

  Shape output_shape{batches, output_height, output_width, output_depth};
  auto output_buf = make_buffer<RET_T, LexicalLayout>(output_shape);

  for (uint32_t batch = 0; batch < batches; ++batch)
  {
    for (uint32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (uint32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (uint32_t out_channel = 0; out_channel < output_depth; ++out_channel)
        {
          const int in_x_origin = (out_x * stride_width) - pad_left;
          const int in_y_origin = (out_y * stride_height) - pad_top;

          RET_T total = static_cast<RET_T>(0);

          for (uint32_t filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            for (uint32_t filter_x = 0; filter_x < filter_width; ++filter_x)
            {
              for (uint32_t in_channel = 0; in_channel < input_depth; ++in_channel)
              {
                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                const int32_t in_y = in_y_origin + dilation_height_factor * filter_y;

                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && ((unsigned)in_x < input_width) && (in_y >= 0) &&
                    ((unsigned)in_y < input_height))
                {
                  auto input_value =
                      input_buf->at(Index({batch, (unsigned)in_y, (unsigned)in_x, in_channel}));
                  auto filter_value =
                      filter_buf->at(Index({out_channel, filter_y, filter_x, in_channel}));
                  total += (input_value * filter_value);
                }
              }
            }
          }
          output_buf.at(Index({batch, out_y, out_x, out_channel})) = total;
        }
      }
    }
  }
  return output_buf;
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::Conv2D *conv2d)
{
  auto ifm_data = annot_data(conv2d->ifm());
  auto ker_data = annot_data(conv2d->ker());

  validate(ifm_data, "Can't find input data of Conv2D");
  validate(ifm_data->shape()->rank() == 4, "ifm rank must be 4");

  validate(ker_data, "Can't find kernel data of Conv2D");
  validate(ker_data->shape()->rank() == 4, "Kernel rank must be 4");

  validate(annot_domain(conv2d->ifm()) == loco::Domain::Feature, "IFM of Conv2D is not feature");
  validate(annot_domain(conv2d->ker()) == loco::Domain::Filter, "Kernel of Conv2D is not filter");

  std::unique_ptr<NodeData> conv2d_result = nullptr;

  if (ifm_data->dtype() == loco::DataType::FLOAT32 && ker_data->dtype() == loco::DataType::FLOAT32)
  {
    auto ifm_buf = ifm_data->as_f32_bufptr();
    auto ker_buf = ker_data->as_f32_bufptr();

    auto conv2d_buf = calc_conv2D<float, float, float>(conv2d, ifm_buf, ker_buf);

    conv2d_result = make_data(conv2d_buf);
  }
  else
    throw std::runtime_error("NYI for these DataTypes");

  assert(conv2d_result != nullptr);

  annot_data(conv2d, std::move(conv2d_result));
  annot_domain(conv2d, loco::Domain::Feature);
}

} // namespace locomotiv
