/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleSVDF.h"

#include <luci/IR/Nodes/CircleSVDF.h>

#include <loco.h>

namespace luci
{

bool CircleSVDFBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  if (!(inputs.size() == 4 || inputs.size() == 5))
    return false;

  return true;
}

CircleNode *CircleSVDFBuilder::build_node(const circle::OperatorT &op,
                                          const std::vector<CircleNode *> &inputs,
                                          loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleSVDF>();
  node->input(inputs.at(0));
  node->weight_feature(inputs.at(1));
  node->weight_time(inputs.at(2));
  if (inputs.size() == 4)
  {
    auto *bias = graph->nodes()->create<CircleOutputExclude>();
    // CircleOutputExclude doesn't need a type, but since all nodes must have a type,
    // a dummy type is inserted.
    bias->dtype(inputs.at(0)->dtype());
    node->bias(bias);

    node->input_activation_state(inputs.at(3));
  }
  else
  {
    node->bias(inputs.at(3));
    node->input_activation_state(inputs.at(4));
  }

  const auto *options = op.builtin_options.AsSVDFOptions();
  node->svdf_rank(options->rank);
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  node->asymmetric_quantize_inputs(options->asymmetric_quantize_inputs);

  return node;
}

} // namespace luci
