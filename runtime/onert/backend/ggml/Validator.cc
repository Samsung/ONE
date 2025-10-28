/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Validator.h"

namespace onert::backend::ggml
{

void Validator::visit(const ir::operation::FullyConnected &node)
{
  using ir::operation::FullyConnected;

  const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto weight_node = &_graph.operands().at(weight_index);

  _supported = false;

  if (weight_node->typeInfo().type() != ir::DataType::QUANT_GGML_Q4_0 &&
      weight_node->typeInfo().type() != ir::DataType::QUANT_GGML_Q8_0)
    return;

  if (node.param().activation != ir::Activation::NONE)
    return;

  _supported = true;
}

void Validator::visit(const ir::operation::Gather &node)
{
  using ir::operation::Gather;

  const auto input_index{node.getInputs().at(Gather::Input::INPUT)};
  const auto input_node = &_graph.operands().at(input_index);

  _supported = false;

  if (input_node->typeInfo().type() != ir::DataType::QUANT_GGML_Q4_0)
    return;

  _supported = true;
}

} // namespace onert::backend::ggml
