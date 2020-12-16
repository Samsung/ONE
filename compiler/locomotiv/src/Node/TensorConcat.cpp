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

namespace
{

using namespace locomotiv;

void execute_node(loco::TensorConcat *tensor_concat)
{
  validate(tensor_concat, "TensorConcat is nullptr");

  auto lhs_data = annot_data(tensor_concat->lhs());
  auto rhs_data = annot_data(tensor_concat->rhs());
  auto axis = tensor_concat->axis();

  validate(lhs_data && rhs_data, "Ingredient not ready");
  validate(lhs_data->dtype() == rhs_data->dtype(), "lhs and rhs of Concat should have same dtype");

  validate(annot_domain(tensor_concat->lhs()) == loco::Domain::Tensor &&
             annot_domain(tensor_concat->rhs()) == loco::Domain::Tensor,
           "Some ingredients of TensorConcat is not Tensor");

  // Calculate output shape
  Shape lhs_shape = *lhs_data->shape();
  Shape rhs_shape = *rhs_data->shape();
  Shape concat_shape;

  assert(lhs_shape.rank() == rhs_shape.rank());
  concat_shape.resize(lhs_shape.rank());
  for (uint32_t index = 0; index < lhs_shape.rank(); ++index)
  {
    if (index == axis)
      concat_shape.dim(index) = lhs_shape.dim(index) + rhs_shape.dim(index);
    else
    {
      assert(lhs_shape.dim(index) == rhs_shape.dim(index));
      concat_shape.dim(index) = lhs_shape.dim(index);
    }
  }
  auto left_dim_size = lhs_shape.dim(axis);

  // Copy data from two inputs LHS and RHS to Concat
  std::unique_ptr<NodeData> concat_data = nullptr;
  switch (lhs_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto lhs_bufptr = lhs_data->as_f32_bufptr();
      auto rhs_bufptr = rhs_data->as_f32_bufptr();
      auto concat_buf = make_buffer<float, LexicalLayout>(concat_shape);

      for (IndexEnumerator e{concat_shape}; e.valid(); e.advance())
      {
        const auto &e_index = e.current();

        if (e_index.at(axis) < left_dim_size)
        {
          // Left index is same as output index
          concat_buf.at(e_index) = lhs_bufptr->at(e_index);
        }
        else
        {
          // Adjust right index to valid range
          Index r_index = e_index;
          r_index.at(axis) -= left_dim_size;
          concat_buf.at(e_index) = rhs_bufptr->at(r_index);
        }
      }

      concat_data = make_data(concat_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(concat_data != nullptr);
  annot_data(tensor_concat, std::move(concat_data));
  annot_domain(tensor_concat, loco::Domain::Tensor);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::TensorConcat *tensor_concat) { execute_node(tensor_concat); }

} // namespace locomotiv
