/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleReshape.h"
#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CircleReshape.h>

#include <oops/UserExn.h>
#include <stdex/Memory.h>

#include <cassert>

namespace luci
{

bool CircleReshapeGraphBuilder::validate(const circle::Operator *op) const
{
  const auto &inputs = *op->inputs();
  const auto &outputs = *op->outputs();

  if (inputs.size() != 1 && inputs.size() != 2)
    return false;

  if (outputs.size() != 1)
    return false;

  return true;
}

void CircleReshapeGraphBuilder::build(const circle::Operator *op,
                                      GraphBuilderContext *context) const
{
  auto graph = context->graph();
  auto reader = context->reader();
  auto nodefinder = context->nodefinder();

  auto tensors = reader->tensors();
  const auto &inputs = *op->inputs();
  const auto &outputs = *op->outputs();

  assert(outputs.size() == 1);
  const circle::Tensor *output_tensor = tensors->Get(outputs[0]);

  // Create the node.
  auto reshape_node = graph->nodes()->create<CircleReshape>();
  reshape_node->name(tensor_name(output_tensor));

  // Set node's quantization parameters, if any.
  auto quantization = tensor_quantization(output_tensor);
  if (quantization)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam)
      reshape_node->quantparam(std::move(quantparam));
  }

  // Set node's inputs. There may be one or two, but the IR requires 2 atm.
  assert(inputs.size() == 1 || inputs.size() == 2);
  if (inputs.size() != 2)
    throw oops::UserExn("Unsupported number of inputs", inputs.size());

  CircleNode *tensor_node = nodefinder->node(inputs[0]);
  assert(tensor_node != nullptr);
  reshape_node->tensor(tensor_node);

  CircleNode *shape_node = nodefinder->node(inputs[1]);
  assert(shape_node != nullptr);
  reshape_node->shape(shape_node);

  // Configure options.
  const circle::ReshapeOptions *options = op->builtin_options_as_ReshapeOptions();
  const auto &new_shape = *options->new_shape();
  reshape_node->newShape()->rank(new_shape.size());
  for (uint32_t i = 0; i < new_shape.size(); ++i)
    reshape_node->newShape()->dim(i) = new_shape[i];

  // Register node's only output.
  nodefinder->enroll(outputs[0], reshape_node);
}

} // namespace luci
