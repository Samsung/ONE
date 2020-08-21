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

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

#include <cassert>
#include <stdexcept>
#include <cmath>

namespace
{

Index reduce_index(const Index &index, uint32_t axis)
{
  Index r_index;

  r_index.resize(index.rank());
  for (uint32_t i = 0; i < index.rank(); ++i)
    r_index.at(i) = index.at(i);
  r_index.at(axis) = 0;

  return r_index;
}

Shape reduce_shape(const Shape &shape, uint32_t axis)
{
  Shape r_shape;

  r_shape.resize(shape.rank());
  for (uint32_t i = 0; i < shape.rank(); ++i)
    r_shape.dim(i) = shape.dim(i);
  r_shape.dim(axis) = 1;

  return r_shape;
}

} // namespace

namespace
{

using namespace locomotiv;

void execute_node(loco::TensorSoftmax *softmax)
{
  auto input_data = annot_data(softmax->input());

  validate(input_data, "Input not ready");
  validate(annot_domain(softmax->input()) == loco::Domain::Tensor,
           "Input domain of TensorSoftmax is not Tensor");

  std::unique_ptr<NodeData> softmax_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto axis = softmax->axis();

      auto *input_shape = input_data->shape();
      auto input_bufptr = input_data->as_f32_bufptr();
      auto softmax_buf = make_buffer<float, LexicalLayout>(*input_data->shape());

      auto reduce_sum_shape = reduce_shape(*input_shape, axis);
      auto reduce_sum_bufptr = make_buffer<float, LexicalLayout>(reduce_sum_shape);

      for (IndexEnumerator e{*input_shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        const auto r_index = reduce_index(index, axis);

        reduce_sum_bufptr.at(r_index) += exp(input_bufptr->at(index));
      }

      for (IndexEnumerator e{*input_shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        const auto r_index = reduce_index(index, axis);

        softmax_buf.at(index) = exp(input_bufptr->at(index)) / reduce_sum_bufptr.at(r_index);
      }

      softmax_data = make_data(softmax_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(softmax_data != nullptr);
  annot_data(softmax, std::move(softmax_data));
  annot_domain(softmax, annot_domain(softmax->input()));
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::TensorSoftmax *softmax) { execute_node(softmax); }

} // namespace locomotiv
