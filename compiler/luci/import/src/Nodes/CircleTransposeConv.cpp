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

#include "luci/Import/Nodes/CircleTransposeConv.h"

#include <luci/IR/Nodes/CircleTransposeConv.h>

#include <loco.h>

#include <cassert>

namespace luci
{

bool CircleTransposeConvGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 3 && args.op.inputs.size() != 4)
    return false;

  const auto &inputs = args.op.inputs;
  const auto tensors = args.reader.tensors();
  const auto filter_tensor = tensors.at(inputs.at(1));
  assert(filter_tensor != nullptr);
  const auto filter_shape = wrap(filter_tensor->shape());
  const auto ifm_tensor = tensors.at(inputs.at(2));
  assert(ifm_tensor != nullptr);
  const auto ifm_shape = wrap(ifm_tensor->shape());

  // ifm and filters must be 4-D tensor
  if (ifm_shape.size() != 4)
    return false;
  if (filter_shape.size() != 4)
    return false;

  // input shape : [batch, height, width, in_channels]
  // filters shape : [output_channels, height, weight, in_channels]
  if (ifm_shape.at(3) != filter_shape.at(3))
    return false;

  return true;
}

CircleNode *CircleTransposeConvGraphBuilder::build_node(const circle::OperatorT &op,
                                                        const std::vector<CircleNode *> &inputs,
                                                        loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleTransposeConv>();

  node->inputSizes(inputs.at(0));
  node->filter(inputs.at(1));
  node->outBackprop(inputs.at(2));
  if (inputs.size() == 3)
  {
    auto *bias = graph->nodes()->create<CircleOutputExclude>();
    node->bias(bias);
  }
  else
    node->bias(inputs.at(3));

  const auto *options = op.builtin_options.AsTransposeConvOptions();
  node->padding(luci_padding(options->padding));
  node->stride()->w(options->stride_w);
  node->stride()->h(options->stride_h);
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));

  return node;
}

} // namespace luci
