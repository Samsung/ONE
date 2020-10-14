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

#include "luci/Import/Nodes/CircleMaxPoolWithArgMax.h"

#include <luci/IR/Nodes/CircleMaxPoolWithArgMax.h>

#include <loco.h>

namespace luci
{

bool CircleMaxPoolWithArgMaxGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 1)
    return false;

  return true;
}

void CircleMaxPoolWithArgMaxGraphBuilder::build(const circle::OperatorT &op,
                                                GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  const auto &opcodes = context->reader()->opcodes();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  auto *node = graph->nodes()->create<CircleMaxPoolWithArgMax>();
  node->input(inputs.at(0));
  
  const auto *options = op.builtin_options.AsMaxPoolWithArgMaxOptions();

  node->padding(luci_padding(options->padding));
  node->stride()->w(options->stride_w);
  node->stride()->h(options->stride_h);
  node->filter()->w(options->filter_width);
  node->filter()->h(options->filter_height);

  assert(outputs.size() > 0);

  
}

} // namespace luci
