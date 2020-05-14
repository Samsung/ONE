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

#include "NodeDomain.h"
#include "NodeDataImpl.h"
#include "Validation.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <cassert>
#include <stdexcept>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

namespace locomotiv
{

float UnaryFunc::apply(float) const { throw std::runtime_error{"F32 is not supported yet"}; }
int32_t UnaryFunc::apply(int32_t) const { throw std::runtime_error{"S32 is not supported yet"}; }

float BinaryFunc::apply(float, float) const
{
  throw std::runtime_error{"F32 is not supported yet"};
}

int32_t BinaryFunc::apply(int32_t, int32_t) const
{
  throw std::runtime_error{"S32 is not supported yet"};
}

// TODO Use visitor pattern of loco when available
void NodeExecution::run(loco::Node *node)
{
  erase_annot_data(node);

#define NODE(Name)                 \
  if (as<loco::Name>(node))        \
  {                                \
    execute(as<loco::Name>(node)); \
    return;                        \
  }
#include "Node.lst"
#undef NODE

  throw std::runtime_error("Not supported loco::Node type");
}

void NodeExecution::eltwise_unary(loco::Node *node, const UnaryFunc &f)
{
  auto input_node = node->arg(0);
  auto input_domain = annot_domain(input_node);
  auto input_data = annot_data(input_node);
  validate(input_data, "Input is not ready");
  auto input_dtype = input_data->dtype();

  validate(input_domain != loco::Domain::Unknown, "Input domain is unknown");

  auto output_node = node;
  // Element-wise Unary Operation does not affect Domain
  auto output_domain = input_domain;
  // Eltwise-wise Unary Operation does not affet Data Type (ASSUMPTION)
  //
  // TODO Check this assumption
  auto output_dtype = input_dtype;
  std::unique_ptr<NodeData> output_data = nullptr;

  switch (output_dtype)
  {
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      auto output_buf = make_buffer<float, LexicalLayout>(*input_data->shape());
      auto *shape = input_data->shape();

      for (IndexEnumerator e{*shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        output_buf.at(index) = f.apply(input_bufptr->at(index));
      }

      output_data = make_data(output_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(output_data != nullptr);
  annot_data(output_node, std::move(output_data));
  annot_domain(output_node, output_domain);
}

void NodeExecution::eltwise_binary(loco::Node *node, const BinaryFunc &f)
{
  auto lhs_node = node->arg(0);
  auto rhs_node = node->arg(1);
  auto lhs_data = annot_data(lhs_node);
  auto rhs_data = annot_data(rhs_node);

  validate(lhs_data && rhs_data, "Input not ready");
  validate(annot_domain(lhs_node) == annot_domain(rhs_node), "Wrong input domain");
  validate(lhs_data->dtype() == rhs_data->dtype(), "Wrong input type");
  validate(*lhs_data->shape() == *rhs_data->shape(), "Wrong input shape");

  auto out_node = node;
  std::unique_ptr<NodeData> out_data = nullptr;

  switch (lhs_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto lhs_bufptr = lhs_data->as_f32_bufptr();
      auto rhs_bufptr = rhs_data->as_f32_bufptr();
      auto out_bufptr = make_buffer<float, LexicalLayout>(*lhs_data->shape());

      auto *shape = lhs_data->shape();

      for (IndexEnumerator e{*shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        out_bufptr.at(index) = f.apply(lhs_bufptr->at(index), rhs_bufptr->at(index));
      }

      out_data = make_data(out_bufptr);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(out_data != nullptr);
  annot_data(out_node, std::move(out_data));
  annot_domain(out_node, annot_domain(lhs_node));
}

} // namespace locomotiv
