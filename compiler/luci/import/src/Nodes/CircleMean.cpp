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

#include "luci/Import/Nodes/CircleMean.h"
#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CircleMean.h>

#include <oops/UserExn.h>
#include <stdex/Memory.h>

#include <cassert>

namespace luci
{

bool CircleMeanGraphBuilder::validate(const circle::Operator *op) const
{
  const auto &inputs = *op->inputs();

  if (inputs.size() != 2)
    return false;

  return true;
}

void CircleMeanGraphBuilder::build(const circle::Operator *op, GraphBuilderContext *context) const
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
  auto mean_node = graph->nodes()->create<CircleMean>();
  mean_node->name(tensor_name(output_tensor));

  // Set node's quantization parameters, if any.
  auto quantization = tensor_quantization(output_tensor);
  if (quantization)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam)
      mean_node->quantparam(std::move(quantparam));
  }

  // input
  CircleNode *input_node = nodefinder->node(inputs[0]);
  assert(input_node != nullptr);
  mean_node->input(input_node);

  // reduction indices
  CircleNode *reduction_insices_node = nodefinder->node(inputs[1]);
  assert(reduction_insices_node != nullptr);
  mean_node->reduction_indices(reduction_insices_node);

  // Configure options.
  const auto *options = op->builtin_options_as_ReducerOptions();
  mean_node->keep_dims(options->keep_dims());

  // Register node's only output.
  nodefinder->enroll(outputs[0], mean_node);
}

} // namespace luci
