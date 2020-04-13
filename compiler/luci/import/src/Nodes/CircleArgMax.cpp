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

#include "luci/Import/Nodes/CircleArgMax.h"
#include "luci/Import/Nodes/CircleConst.h"
#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CircleArgMax.h>
#include <luci/Log.h>

#include <loco.h>
#include <stdex/Memory.h>

#include <cassert>

namespace
{

using namespace luci;

class CircleArgMaxGraphUpdate final : public GraphUpdate
{
public:
  CircleArgMaxGraphUpdate(CircleArgMax *node) : _node(node) {}

  void update(GraphBuilderContext *) override;

private:
  CircleArgMax *_node;
};

} // namespace

namespace luci
{

bool CircleArgMaxGraphBuilder::validate(const circle::Operator *op) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  if (inputs.size() != 2)
    return false;

  return true;
}

void CircleArgMaxGraphBuilder::build(const circle::Operator *op, GraphBuilderContext *context) const
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
  assert(outputs.size() > 0);

  // ArgMax node itself
  auto argmax_node = graph->nodes()->create<CircleArgMax>();
  uint32_t output_ti = static_cast<uint32_t>(outputs[0]);
  auto output_tensor = tensors->Get(output_ti);
  auto tname = tensor_name(output_tensor);
  argmax_node->name(tname);
  opfinder->enroll(argmax_node, op);
  tensorfinder->enroll(argmax_node, output_tensor);
  for (auto output : outputs)
  {
    INFO(l) << "[luci] NodeFinder argmax_node(" << output << ") -> " << argmax_node << std::endl;
    nodefinder->enroll(output, argmax_node);
  }
  const auto *options = op->builtin_options_as_ArgMaxOptions();
  if (options != nullptr)
  {
    // output_type
    auto output_type = luci_datatype(options->output_type());
    argmax_node->output_type(output_type);
  }

  // ArgMax dimension tensor + buffer to CircleConst node
  uint32_t dimension_ti = static_cast<uint32_t>(inputs[1]);
  auto dimension_const = create_circleconst(context, dimension_ti);
  argmax_node->dimension(dimension_const);

  // Create GraphUpdate for graph connection for Add node
  auto update = stdex::make_unique<CircleArgMaxGraphUpdate>(argmax_node);
  updates->enroll(std::move(update));
}

} // namespace luci

namespace
{

void CircleArgMaxGraphUpdate::update(GraphBuilderContext *context)
{
  LOGGER(l);

  auto opfinder = context->opfinder();
  auto nodefinder = context->nodefinder();

  auto op = opfinder->op(_node);

  // set input
  const std::vector<int32_t> &inputs = luci::as_index_vector(op->inputs());
  uint32_t idx_0 = static_cast<uint32_t>(inputs[0]);
  uint32_t idx_1 = static_cast<uint32_t>(inputs[1]);
  INFO(l) << "[luci] ArgMax update " << idx_0 << ", " << idx_1 << std::endl;
  auto node_0 = nodefinder->node(idx_0);
  assert(node_0 != nullptr);
  auto node_1 = nodefinder->node(idx_1);
  (void)node_1; // unused error for release build
  assert(node_1 != nullptr);
  _node->input(node_0);
  assert(_node->dimension() == node_1);
}

} // namespace
