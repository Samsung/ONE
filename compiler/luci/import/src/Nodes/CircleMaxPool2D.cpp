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

#include "luci/Import/Nodes/CircleMaxPool2D.h"
#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CircleAdd.h>
#include <luci/Log.h>

#include <loco.h>
#include <stdex/Memory.h>

#include <cassert>

namespace
{

using namespace luci;

class CircleMaxPool2DGraphUpdate final : public GraphUpdate
{
public:
  CircleMaxPool2DGraphUpdate(CircleMaxPool2D *node) : _node(node) {}

  void update(GraphBuilderContext *) override;

private:
  CircleMaxPool2D *_node;
};

} // namespace

namespace luci
{

bool CircleMaxPool2DGraphBuilder::validate(const circle::Operator *op) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  if (inputs.size() != 1)
    return false;

  return true;
}

void CircleMaxPool2DGraphBuilder::build(const circle::Operator *op,
                                        GraphBuilderContext *context) const
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

  // MaxPool2D node itself
  auto maxpool2d_node = graph->nodes()->create<CircleMaxPool2D>();
  assert(outputs.size() > 0);
  uint32_t output_ti = static_cast<uint32_t>(outputs[0]);
  auto output_tensor = tensors->Get(output_ti);

  auto tname = tensor_name(output_tensor);
  maxpool2d_node->name(tname);
  auto quantization = tensor_quantization(output_tensor);
  if (quantization)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam.get())
      maxpool2d_node->quantparam(std::move(quantparam));
  }

  opfinder->enroll(maxpool2d_node, op);
  tensorfinder->enroll(maxpool2d_node, output_tensor);
  for (auto output : outputs)
  {
    INFO(l) << "[luci] NodeFinder maxpool2d_node(" << output << ") -> " << maxpool2d_node
            << std::endl;
    nodefinder->enroll(output, maxpool2d_node);
  }
  const auto *options = op->builtin_options_as_Pool2DOptions();

  // Filter
  maxpool2d_node->filter()->w(options->filter_width());
  maxpool2d_node->filter()->h(options->filter_height());

  // Padding
  auto padding = luci_padding(options->padding());
  maxpool2d_node->padding(padding);

  // Stride
  maxpool2d_node->stride()->w(options->stride_w());
  maxpool2d_node->stride()->h(options->stride_h());

  // Activation
  auto actfunctype = luci_actfunc(options->fused_activation_function());
  maxpool2d_node->fusedActivationFunction(actfunctype);

  // Create GraphUpdate for graph connection for MaxPool2D node
  auto update = stdex::make_unique<CircleMaxPool2DGraphUpdate>(maxpool2d_node);
  updates->enroll(std::move(update));
}

} // namespace luci

namespace
{

void CircleMaxPool2DGraphUpdate::update(GraphBuilderContext *context)
{
  auto opfinder = context->opfinder();
  auto nodefinder = context->nodefinder();

  auto op = opfinder->op(_node);

  // set input 'value'
  const std::vector<int32_t> &inputs = luci::as_index_vector(op->inputs());
  uint32_t idx_value = static_cast<uint32_t>(inputs[0]);
  auto node_value = nodefinder->node(idx_value);
  assert(node_value != nullptr);
  _node->value(node_value);
}

} // namespace
