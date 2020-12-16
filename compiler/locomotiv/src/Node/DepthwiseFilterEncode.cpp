/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <stdexcept>
#include <cassert>

namespace
{

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::IndexEnumerator;

/**
 * @brief  Encode input tensor into depthwise filter represented in "HWCM" layout
 *
 * (Please check locomotiv README for further information)
 */
template <typename T>
std::unique_ptr<locomotiv::NodeData> dw_filter_encode(const loco::DepthwiseFilterEncode *node,
                                                      const Buffer<T> *input_buf)
{
  auto encoder = node->encoder();

  // Make TensorShape from input
  loco::TensorShape input_shape;
  input_shape.rank(input_buf->shape().rank());
  assert(input_shape.rank() == 4);
  for (uint32_t i = 0; i < input_shape.rank(); ++i)
  {
    input_shape.dim(i) = input_buf->shape().dim(i);
  }

  loco::DepthwiseFilterShape node_shape = encoder->shape(input_shape);

  // Make HWCM (i.e. height, width, depth, multiplier) buffer from DepthwiseFilterShape
  Buffer<T> node_buf = make_buffer<T, LexicalLayout>(
    Shape{node_shape.height().value(), node_shape.width().value(), node_shape.depth().value(),
          node_shape.multiplier().value()});

  // Copy buffer in an order arranged by encoder
  for (IndexEnumerator e{node_buf.shape()}; e.valid(); e.advance())
  {
    loco::DepthwiseFilterIndex index;
    index.row() = e.current().at(0);
    index.column() = e.current().at(1);
    index.channel() = e.current().at(2);
    index.nth() = e.current().at(3);

    node_buf.at(e.current()) = input_buf->at(encoder->value(index));
  }

  return locomotiv::make_data(node_buf);
}

} // namespace

namespace
{

using namespace locomotiv;

void execute_node(loco::DepthwiseFilterEncode *enc)
{
  auto input_data = annot_data(enc->input());

  validate(input_data, "Input of DepthwiseFilterEncode not ready");
  validate(annot_domain(enc->input()) == loco::Domain::Tensor,
           "Input of DepthwiseFilterEncode is not Tensor");
  validate(input_data->shape()->rank() == 4, "Input shape mismatch");

  std::unique_ptr<NodeData> enc_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto input_buf = input_data->as_f32_bufptr();
      enc_data = dw_filter_encode<float>(enc, input_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(enc_data != nullptr);
  annot_data(enc, std::move(enc_data));
  annot_domain(enc, loco::Domain::DepthwiseFilter);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::DepthwiseFilterEncode *enc) { execute_node(enc); }

} // namespace locomotiv
