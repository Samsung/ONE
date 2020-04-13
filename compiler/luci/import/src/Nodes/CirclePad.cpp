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

#include "luci/Import/Nodes/CirclePad.h"
#include "luci/Import/Nodes/CircleConst.h"
#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CirclePad.h>
#include <luci/Log.h>

#include <loco.h>
#include <stdex/Memory.h>

#include <cassert>

namespace
{

using namespace luci;

class CirclePadGraphUpdate final : public GraphUpdate
{
public:
  CirclePadGraphUpdate(CirclePad *node) : _node(node) {}

  void update(GraphBuilderContext *) override;

private:
  CirclePad *_node;
};

} // namespace

namespace luci
{

bool CirclePadGraphBuilder::validate(const circle::Operator *op) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  if (inputs.size() != 2)
    return false;

  // TODO do attribute checks

  return true;
}

void CirclePadGraphBuilder::build(const circle::Operator *op, GraphBuilderContext *context) const
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

  // Pad node itself
  auto pad_node = graph->nodes()->create<CirclePad>();
  assert(outputs.size() > 0);
  uint32_t output_ti = static_cast<uint32_t>(outputs[0]);
  auto output_tensor = tensors->Get(output_ti);

  // name
  auto tname = tensor_name(output_tensor);
  pad_node->name(tname);

  // quantization
  auto quantization = tensor_quantization(output_tensor);
  if (quantization)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam.get())
      pad_node->quantparam(std::move(quantparam));
  }

  opfinder->enroll(pad_node, op);
  tensorfinder->enroll(pad_node, output_tensor);
  for (auto output : outputs)
  {
    INFO(l) << "[luci] NodeFinder pad_node(" << output << ") -> " << pad_node << std::endl;
    nodefinder->enroll(output, pad_node);
  }

  // There's no options to read for Pad

  // paddings Const
  uint32_t paddings_ti = static_cast<uint32_t>(inputs[1]);
  auto paddings_const = create_circleconst(context, paddings_ti);
  pad_node->paddings(paddings_const);

  // Create GraphUpdate for graph connection for Pad node
  auto update = stdex::make_unique<CirclePadGraphUpdate>(pad_node);
  updates->enroll(std::move(update));
}

} // namespace luci

namespace
{

void CirclePadGraphUpdate::update(GraphBuilderContext *context)
{
  auto opfinder = context->opfinder();
  auto nodefinder = context->nodefinder();

  auto op = opfinder->op(_node);

  // set input 'input, paddings'
  const std::vector<int32_t> &inputs = luci::as_index_vector(op->inputs());
  uint32_t idx_input = static_cast<uint32_t>(inputs[0]);
  auto node_input = nodefinder->node(idx_input);
  assert(node_input != nullptr);
  _node->input(node_input);
  // paddings CircleConst is created in build() and should not be null
  assert(_node->paddings() != nullptr);
}

} // namespace
