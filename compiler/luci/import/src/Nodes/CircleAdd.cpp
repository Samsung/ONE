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

#include "luci/Import/Nodes/CircleAdd.h"
#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CircleAdd.h>
#include <luci/Log.h>

#include <loco.h>
#include <stdex/Memory.h>

#include <cassert>

namespace
{

using namespace luci;

class CircleAddGraphUpdate final : public GraphUpdate
{
public:
  CircleAddGraphUpdate(CircleAdd *node) : _node(node) {}

  void update(GraphBuilderContext *) override;

private:
  CircleAdd *_node;
};

} // namespace

namespace luci
{

bool CircleAddGraphBuilder::validate(const circle::Operator *op) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  if (inputs.size() != 2)
    return false;

  return true;
}

void CircleAddGraphBuilder::build(const circle::Operator *op, GraphBuilderContext *context) const
{
  LOGGER(l);

  assert(context != nullptr);

  auto graph = context->graph();
  auto reader = context->reader();
  auto opfinder = context->opfinder();
  auto tensorfinder = context->tensorfinder();
  auto nodefinder = context->nodefinder();
  auto updates = context->updates();

  // FlatBuffer contents
  auto tensors = reader->tensors();

  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  const std::vector<int32_t> &outputs = as_index_vector(op->outputs());

  // Add node itself
  auto add_node = graph->nodes()->create<CircleAdd>();
  assert(outputs.size() > 0);
  uint32_t output_ti = static_cast<uint32_t>(outputs[0]);
  auto output_tensor = tensors->Get(output_ti);
  auto tname = tensor_name(output_tensor);
  add_node->name(tname);
  auto quantization = tensor_quantization(output_tensor);
  if (quantization)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam.get())
      add_node->quantparam(std::move(quantparam));
  }
  opfinder->enroll(add_node, op);
  tensorfinder->enroll(add_node, output_tensor);
  for (auto output : outputs)
  {
    INFO(l) << "[luci] NodeFinder add_node(" << output << ") -> " << add_node << std::endl;
    nodefinder->enroll(output, add_node);
  }
  const auto *options = op->builtin_options_as_AddOptions();

  // Activation
  auto actfunctype = luci_actfunc(options->fused_activation_function());
  add_node->fusedActivationFunction(actfunctype);

  // Create GraphUpdate for graph connection for Add node
  auto update = stdex::make_unique<CircleAddGraphUpdate>(add_node);
  updates->enroll(std::move(update));
}

} // namespace luci

namespace
{

void CircleAddGraphUpdate::update(GraphBuilderContext *context)
{
  auto opfinder = context->opfinder();
  auto nodefinder = context->nodefinder();

  auto op = opfinder->op(_node);

  // set input 'x, y'
  const std::vector<int32_t> &inputs = luci::as_index_vector(op->inputs());
  uint32_t idx_x = static_cast<uint32_t>(inputs[0]);
  uint32_t idx_y = static_cast<uint32_t>(inputs[1]);
  auto node_x = nodefinder->node(idx_x);
  assert(node_x != nullptr);
  auto node_y = nodefinder->node(idx_y);
  _node->x(node_x);
  _node->y(node_y);
}

} // namespace
