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

template <typename T>
std::unique_ptr<locomotiv::NodeData> matrix_encode(const loco::MatrixEncode *node,
                                                   const Buffer<T> *input_buf)
{
  auto encoder = node->encoder();

  // Make TensorShape from input
  loco::TensorShape input_shape;
  input_shape.rank(input_buf->shape().rank());
  assert(input_shape.rank() == 2);
  for (uint32_t i = 0; i < input_shape.rank(); ++i)
  {
    input_shape.dim(i) = input_buf->shape().dim(i);
  }

  loco::MatrixShape node_shape = encoder->shape(input_shape);

  // Make HW buffer from MatrixShape
  Buffer<T> node_buf =
      make_buffer<T, LexicalLayout>(Shape{node_shape.height().value(), node_shape.width().value()});

  // Copy buffer in an order arranged by encoder
  for (IndexEnumerator e{node_buf.shape()}; e.valid(); e.advance())
  {
    loco::MatrixIndex index;
    index.row() = e.current().at(0);
    index.column() = e.current().at(1);

    node_buf.at(e.current()) = input_buf->at(encoder->value(index));
  }

  return locomotiv::make_data(node_buf);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::MatrixEncode *matrix_enc)
{
  auto input_data = annot_data(matrix_enc->input());

  validate(input_data, "Input not ready");
  validate(annot_domain(matrix_enc->input()) == loco::Domain::Tensor,
           "Input domain should be Tensor");
  validate(input_data->shape()->rank() == 2, "Input data rank must be 2");

  std::unique_ptr<NodeData> matrix_enc_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::S32:
    {
      auto input_buf = input_data->as_s32_bufptr();
      matrix_enc_data = matrix_encode<int32_t>(matrix_enc, input_buf);
      break;
    }
    case loco::DataType::FLOAT32:
    {
      auto input_buf = input_data->as_f32_bufptr();
      matrix_enc_data = matrix_encode<float>(matrix_enc, input_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(matrix_enc_data != nullptr);

  annot_data(matrix_enc, std::move(matrix_enc_data));
  annot_domain(matrix_enc, loco::Domain::Matrix);
}

} // namespace locomotiv
