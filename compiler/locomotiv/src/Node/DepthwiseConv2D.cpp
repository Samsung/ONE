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

/**
 * @brief Calculates DepthwiseConv2D
 * @note  ifm_buf has NHWC and ker_buf HWCM format
 *        (Please check locomotiv README for further information)
 */
template <typename RET_T, typename IFM_T, typename KER_T>
Buffer<RET_T> calc_dw_conv2d(const loco::DepthwiseConv2D *dw_conv2d, const Buffer<IFM_T> *ifm_buf,
                             const Buffer<KER_T> *ker_buf)
{
  auto ifm_shape = ifm_buf->shape();
  auto ker_shape = ker_buf->shape();

  locomotiv::validate(ifm_shape.rank() == 4, "ifm rank must be 4");
  locomotiv::validate(ker_shape.rank() == 4, "depthwise filter rank must be 4");
  locomotiv::validate(ifm_shape.dim(3 /* of NHWC */) == ker_shape.dim(2 /* of HWCM */),
                      "channel value mismatch"); // should have same channel values

  const uint32_t ifm_height = ifm_shape.dim(1);
  const uint32_t ifm_width = ifm_shape.dim(2);

  const uint32_t ker_height = ker_shape.dim(0);
  const uint32_t ker_width = ker_shape.dim(1);

  const uint32_t stride_width = dw_conv2d->stride()->horizontal();
  const uint32_t stride_height = dw_conv2d->stride()->vertical();

  // TODO Enable dilations. Let's set these to 1 for now.
  const uint32_t dilation_width_factor = 1;
  const uint32_t dilation_height_factor = 1;

  const uint32_t pad_top = dw_conv2d->pad()->top();
  const uint32_t pad_bottom = dw_conv2d->pad()->bottom();

  const uint32_t pad_left = dw_conv2d->pad()->left();
  const uint32_t pad_right = dw_conv2d->pad()->right();

  const uint32_t ofm_height =
    compute_out_size(ifm_height, pad_top + pad_bottom, ker_height, stride_height);
  const uint32_t ofm_width =
    compute_out_size(ifm_width, pad_left + pad_right, ker_width, stride_width);

  const uint32_t batches = ifm_shape.dim(0);
  const uint32_t ifm_depth = ifm_shape.dim(3);
  const uint32_t multiplier = ker_shape.dim(3);
  const uint32_t ofm_depth = ifm_depth * multiplier;

  Shape ofm_shape{batches, ofm_height, ofm_width, ofm_depth};
  auto ofm_buf = make_buffer<RET_T, LexicalLayout>(ofm_shape);

  for (uint32_t batch = 0; batch < batches; ++batch)
  {
    for (uint32_t ofm_y = 0; ofm_y < ofm_height; ++ofm_y)
    {
      for (uint32_t ofm_x = 0; ofm_x < ofm_width; ++ofm_x)
      {
        for (uint32_t ch = 0; ch < ifm_depth; ++ch)
        {
          for (uint32_t nth = 0; nth < multiplier; nth++)
          {
            const int in_x_origin = (ofm_x * stride_width) - pad_left;
            const int in_y_origin = (ofm_y * stride_height) - pad_top;
            float total = 0.f;
            for (uint32_t ker_y = 0; ker_y < ker_height; ++ker_y)
            {
              for (uint32_t ker_x = 0; ker_x < ker_width; ++ker_x)
              {
                const int in_x = in_x_origin + dilation_width_factor * ker_x;
                const int in_y = in_y_origin + dilation_height_factor * ker_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && ((unsigned)in_x < ifm_width) && (in_y >= 0) &&
                    ((unsigned)in_y < ifm_height))
                {
                  auto ifm_value = ifm_buf->at(Index({batch, (unsigned)in_y, (unsigned)in_x, ch}));
                  auto ker_value = ker_buf->at(Index({ker_y, ker_x, ch, nth}));
                  total += (ifm_value * ker_value);
                }
              }
            }
            uint32_t ofm_channel = ch * multiplier + nth;
            ofm_buf.at(Index({batch, ofm_y, ofm_x, ofm_channel})) = total;
          }
        }
      }
    }
  }
  return ofm_buf;
}

} // namespace

namespace
{

using namespace locomotiv;

void execute_node(loco::DepthwiseConv2D *dw_conv2d)
{
  auto ifm_data = annot_data(dw_conv2d->ifm());
  auto ker_data = annot_data(dw_conv2d->ker());

  validate(ifm_data, "Can't find input data of DepthwiseConv2D");
  validate(ifm_data->shape()->rank() == 4, "ifm rank must be 4");

  validate(ker_data, "Can't find kernel data of DepthwiseConv2D");
  validate(ker_data->shape()->rank() == 4, "Kernel rank must be 4");

  validate(annot_domain(dw_conv2d->ifm()) == loco::Domain::Feature,
           "IFM of DepthwiseConv2D is not feature");
  validate(annot_domain(dw_conv2d->ker()) == loco::Domain::DepthwiseFilter,
           "Kernel of DepthwiseConv2D is not depthwise filter");

  std::unique_ptr<NodeData> dw_conv2d_result = nullptr;

  if (ifm_data->dtype() == loco::DataType::FLOAT32 && ker_data->dtype() == loco::DataType::FLOAT32)
  {
    auto ifm_buf = ifm_data->as_f32_bufptr();
    auto ker_buf = ker_data->as_f32_bufptr();

    auto dw_conv2d_buf = calc_dw_conv2d<float, float, float>(dw_conv2d, ifm_buf, ker_buf);

    dw_conv2d_result = make_data(dw_conv2d_buf);
  }
  else
    throw std::runtime_error("NYI for these DataTypes");

  assert(dw_conv2d_result != nullptr);

  annot_data(dw_conv2d, std::move(dw_conv2d_result));
  annot_domain(dw_conv2d, loco::Domain::Feature);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::DepthwiseConv2D *dw_conv2d) { execute_node(dw_conv2d); }

} // namespace locomotiv
