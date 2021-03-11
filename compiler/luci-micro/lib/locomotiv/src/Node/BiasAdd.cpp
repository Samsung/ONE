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
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

#include <cassert>
#include <stdexcept>

namespace
{
using locomotiv::NodeData;

std::unique_ptr<NodeData> calc(const NodeData *input_data, const NodeData *bias_data,
                               uint32_t axis);

} // namespace

namespace
{

using namespace locomotiv;

void execute_node(loco::BiasAdd<loco::Domain::Tensor> *bias_add)
{
  validate(bias_add, "BiasAdd is nullptr");

  auto input_data = locomotiv::annot_data(bias_add->value());
  auto bias_data = locomotiv::annot_data(bias_add->bias());

  validate(input_data && bias_data, "Input not ready");
  validate(locomotiv::annot_domain(bias_add->value()) == loco::Domain::Tensor &&
               locomotiv::annot_domain(bias_add->bias()) == loco::Domain::Bias,
           "Wrong input domain");

  std::unique_ptr<NodeData> bias_add_data = calc(input_data, bias_data, bias_add->axis());

  assert(bias_add_data != nullptr);
  annot_data(bias_add, std::move(bias_add_data));
  annot_domain(bias_add, annot_domain(bias_add->value()));
}

void execute_node(loco::BiasAdd<loco::Domain::Feature> *bias_add)
{
  validate(bias_add, "BiasAdd is nullptr");

  auto input_data = locomotiv::annot_data(bias_add->value());
  auto bias_data = locomotiv::annot_data(bias_add->bias());

  validate(input_data && bias_data, "Input not ready");
  validate(locomotiv::annot_domain(bias_add->value()) == loco::Domain::Feature &&
               locomotiv::annot_domain(bias_add->bias()) == loco::Domain::Bias,
           "Wrong input domain");

  std::unique_ptr<NodeData> bias_add_data = calc(input_data, bias_data, 3);

  assert(bias_add_data != nullptr);
  annot_data(bias_add, std::move(bias_add_data));
  annot_domain(bias_add, loco::Domain::Feature);
}

} // namespace

namespace
{
using locomotiv::NodeData;
using locomotiv::validate;
using locomotiv::make_data;

std::unique_ptr<NodeData> calc(const NodeData *input_data, const NodeData *bias_data, uint32_t axis)
{
  validate(input_data->shape()->dim(axis) == bias_data->shape()->dim(0), "Bias size mismatch");

  std::unique_ptr<NodeData> bias_add_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      auto bias_bufptr = bias_data->as_f32_bufptr();
      auto bias_add_buf = make_buffer<float, LexicalLayout>(*input_data->shape());

      auto *shape = input_data->shape();

      for (IndexEnumerator e{*shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        nncc::core::ADT::tensor::Index bias_index({index.at(axis)});
        bias_add_buf.at(index) = input_bufptr->at(index) + bias_bufptr->at(bias_index);
      }

      bias_add_data = make_data(bias_add_buf);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }
  return bias_add_data;
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::BiasAdd<loco::Domain::Tensor> *bias_add)
{
  execute_node(bias_add);
}

void NodeExecution::execute(loco::BiasAdd<loco::Domain::Feature> *bias_add)
{
  execute_node(bias_add);
}

} // namespace locomotiv
