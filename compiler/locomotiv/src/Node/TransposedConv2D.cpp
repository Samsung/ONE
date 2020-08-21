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

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

/**
 * @brief Compute 1D output size for transposed convolution based on given 1D arguments.
 *
 * @param whole_pad  Sum of front and rear pad
 */
inline uint32_t compute_transposed_out_size(uint32_t input_size, uint32_t whole_pad,
                                            uint32_t filter_size, uint32_t stride)
{
  return stride * (input_size - 1) + filter_size - whole_pad;
}

/**
 * @brief Calculates TransposedConv2D
 * @note  Both input_buf and filter_buf have NHWC format
 */
template <typename RET_T, typename IFM_T, typename FIL_T>
Buffer<RET_T> calc_tr_conv2D(const loco::TransposedConv2D *tr_conv2d,
                             const Buffer<IFM_T> *input_buf, const Buffer<FIL_T> *filter_buf)
{
  auto input_shape = input_buf->shape();
  auto filter_shape = filter_buf->shape();

  locomotiv::validate(input_shape.rank() == 4, "ifm rank must be 4");
  locomotiv::validate(filter_shape.rank() == 4, "filter rank must be 4");
  locomotiv::validate(input_shape.dim(3) /* depth of input */ ==
                          filter_shape.dim(3) /* depth of filter */,
                      "channel value mismatch");

  const uint32_t input_height = input_shape.dim(1);
  const uint32_t input_width = input_shape.dim(2);

  const uint32_t filter_height = filter_shape.dim(1);
  const uint32_t filter_width = filter_shape.dim(2);

  const uint32_t stride_width = tr_conv2d->stride()->horizontal();
  const uint32_t stride_height = tr_conv2d->stride()->vertical();

  const uint32_t pad_top = tr_conv2d->pad()->top();
  const uint32_t pad_bottom = tr_conv2d->pad()->bottom();

  const uint32_t pad_left = tr_conv2d->pad()->left();
  const uint32_t pad_right = tr_conv2d->pad()->right();

  // TODO Support dilations

  const uint32_t output_height =
      compute_transposed_out_size(input_height, pad_top + pad_bottom, filter_height, stride_height);
  const uint32_t output_width =
      compute_transposed_out_size(input_width, pad_left + pad_right, filter_width, stride_width);

  const uint32_t batches = input_shape.dim(0);
  const uint32_t input_depth = input_shape.dim(3);
  const uint32_t output_depth = filter_shape.dim(0); // count of filter

  Shape output_shape{batches, output_height, output_width, output_depth};
  auto output_buf = make_buffer<RET_T, LexicalLayout>(output_shape);

  // initialize output
  for (IndexEnumerator e{output_shape}; e.valid(); e.advance())
  {
    const auto &index = e.current();
    output_buf.at(index) = static_cast<RET_T>(0);
  }

  // Loop through input elements one at a time.
  for (uint32_t batch = 0; batch < batches; ++batch)
  {
    for (uint32_t in_y = 0; in_y < input_height; ++in_y)
    {
      for (uint32_t in_x = 0; in_x < input_width; ++in_x)
      {
        for (uint32_t in_channel = 0; in_channel < input_depth; ++in_channel)
        {
          // Loop through the output elements it will influence
          const int out_x_origin = (in_x * stride_width) - pad_left;
          const int out_y_origin = (in_y * stride_height) - pad_top;
          for (uint32_t filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            for (uint32_t filter_x = 0; filter_x < filter_width; ++filter_x)
            {
              for (uint32_t out_channel = 0; out_channel < output_depth; ++out_channel)
              {
                // Compute output element location
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds
                if ((out_x >= 0) && ((unsigned)out_x < output_width) && (out_y >= 0) &&
                    ((unsigned)out_y < output_height))
                {
                  auto input_value = input_buf->at(Index({batch, in_y, in_x, in_channel}));
                  auto filter_value =
                      filter_buf->at(Index({out_channel, filter_y, filter_x, in_channel}));
                  output_buf.at(Index({batch, (unsigned)out_y, (unsigned)out_x, out_channel})) +=
                      input_value * filter_value;
                }
              }
            }
          }
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

void execute_node(loco::TransposedConv2D *tr_conv2d)
{
  auto ifm_data = annot_data(tr_conv2d->ifm());
  auto ker_data = annot_data(tr_conv2d->ker());

  validate(ifm_data, "Can't find input data of TransposedConv2D");
  validate(ifm_data->shape()->rank() == 4, "ifm rank must be 4");

  validate(ker_data, "Can't find kernel data of TransposedConv2D");
  validate(ker_data->shape()->rank() == 4, "Kernel rank must be 4");

  validate(annot_domain(tr_conv2d->ifm()) == loco::Domain::Feature,
           "IFM of TransposedConv2D is not feature");
  validate(annot_domain(tr_conv2d->ker()) == loco::Domain::Filter,
           "Kernel of TransposedConv2D is not filter");

  std::unique_ptr<NodeData> tr_conv2d_result = nullptr;

  if (ifm_data->dtype() == loco::DataType::FLOAT32 && ker_data->dtype() == loco::DataType::FLOAT32)
  {
    auto ifm_buf = ifm_data->as_f32_bufptr();
    auto ker_buf = ker_data->as_f32_bufptr();

    auto tr_conv2d_buf = calc_tr_conv2D<float, float, float>(tr_conv2d, ifm_buf, ker_buf);

    tr_conv2d_result = make_data(tr_conv2d_buf);
  }
  else
    throw std::runtime_error("NYI for these DataTypes");

  assert(tr_conv2d_result != nullptr);

  annot_data(tr_conv2d, std::move(tr_conv2d_result));
  annot_domain(tr_conv2d, loco::Domain::Feature);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::TransposedConv2D *tr_conv2d) { execute_node(tr_conv2d); }

} // namespace locomotiv
