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

#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <cassert>

using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

namespace
{

using namespace locomotiv;

void execute_node(loco::TensorConstantPad *pad)
{
  validate(pad, "TensorConstantPad is nullptr");

  auto input_data = annot_data(pad->input());
  auto input_domain = annot_domain(pad->input());
  validate(input_data, "Input not ready");
  validate(input_domain == loco::Domain::Tensor, "Input domain of TensorConstantPad is not Tensor");

  auto input_shape = input_data->shape();
  const uint32_t input_rank = input_shape->rank();

  auto padding = pad->padding();
  validate(input_rank == padding->rank(), "input and padding should have same rank");

  auto constant_node = pad->constant();
  auto constant_data = annot_data(constant_node);
  validate(constant_data->dtype() == input_data->dtype(), "constant and input have same data type");
  validate(constant_data->shape()->rank() == 1 && constant_data->shape()->dim(0) == 1,
           "constant should have one rank with one dimension at zero axis");

  std::unique_ptr<NodeData> pad_data = nullptr;
  Index base_index;
  base_index.resize(input_rank);

  // Tensor is padded by relocating its base.
  // padded output index = input index + base index
  for (uint32_t axis = 0; axis < padding->rank(); axis++)
  {
    base_index.at(axis) = padding->front(axis);
  }

  // calculate output shape
  Shape output_shape;
  output_shape.resize(input_rank);
  for (uint32_t i = 0; i < input_rank; i++)
  {
    output_shape.dim(i) = input_shape->dim(i) + padding->front(i) + padding->back(i);
  }

  switch (input_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto input_buf = input_data->as_f32_bufptr();
      auto constant_data_buf = constant_data->as_f32_bufptr();
      const auto constant_value = constant_data_buf->at(Index{0});

      auto output_buf = make_buffer<float, LexicalLayout>(output_shape);

      for (IndexEnumerator ie{*input_shape}, oe{output_shape}; oe.valid(); oe.advance())
      {
        auto input_index = ie.current();
        auto output_index = oe.current();

        if ((input_index + base_index) == output_index)
        {
          output_buf.at(output_index) = input_buf->at(input_index);
          ie.advance();
        }
        else
        {
          output_buf.at(output_index) = constant_value;
        }
      }

      pad_data = make_data(output_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(pad_data != nullptr);
  annot_data(pad, std::move(pad_data));
  annot_domain(pad, annot_domain(pad->input()));
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::TensorConstantPad *pad) { execute_node(pad); }

} // namespace locomotiv
