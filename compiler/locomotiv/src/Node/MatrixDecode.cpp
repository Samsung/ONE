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
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

template <typename T>
std::unique_ptr<locomotiv::NodeData> matrix_decode(const loco::MatrixDecode *node,
                                                   const Buffer<T> *input_buf)
{
  auto decoder = node->decoder();

  // Make MatrixShape from input. Note that matrix in locomotiv represented as HW
  loco::MatrixShape input_shape;
  assert(input_buf->shape().rank() == 2);
  input_shape.height() = input_buf->shape().dim(0);
  input_shape.width() = input_buf->shape().dim(1);

  loco::TensorShape node_shape = decoder->shape(input_shape);

  // Make tensor buffer from TensorShape
  Buffer<T> node_buf =
      make_buffer<T, LexicalLayout>(Shape{node_shape.dim(0).value(), node_shape.dim(1).value()});

  // Copy buffer in an order arranged by decoder
  for (IndexEnumerator e{node_buf.shape()}; e.valid(); e.advance())
  {
    loco::MatrixIndex matrix_index = decoder->value(e.current());
    Index buf_index({matrix_index.row(), matrix_index.column()});

    node_buf.at(e.current()) = input_buf->at(buf_index);
  }

  return locomotiv::make_data(node_buf);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::MatrixDecode *matrix_dec)
{
  auto input_data = annot_data(matrix_dec->input());

  validate(input_data, "Input not ready");
  validate(annot_domain(matrix_dec->input()) == loco::Domain::Matrix,
           "Input domain should be Matrix");
  validate(input_data->shape()->rank() == 2, "Input data rank must be 2");

  std::unique_ptr<NodeData> matrix_dec_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::S32:
    {
      auto input_buf = input_data->as_s32_bufptr();
      matrix_dec_data = matrix_decode<int32_t>(matrix_dec, input_buf);
      break;
    }
    case loco::DataType::FLOAT32:
    {
      auto input_buf = input_data->as_f32_bufptr();
      matrix_dec_data = matrix_decode<float>(matrix_dec, input_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(matrix_dec_data != nullptr);

  annot_data(matrix_dec, std::move(matrix_dec_data));
  annot_domain(matrix_dec, loco::Domain::Tensor);
}

} // namespace locomotiv
