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

#include "luci/Import/Nodes/CircleConv2D.h"
#include "luci/Import/Nodes/CircleConst.h"
#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CircleConv2D.h>
#include <luci/IR/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <loco.h>
#include <stdex/Memory.h>

#include <cassert>

namespace
{

using namespace luci;

class CircleConv2DGraphUpdate final : public GraphUpdate
{
public:
  CircleConv2DGraphUpdate(CircleConv2D *node) : _node(node) {}

  void update(GraphBuilderContext *) override;

private:
  CircleConv2D *_node;
};

} // namespace

namespace luci
{

bool CircleConv2DGraphBuilder::validate(const circle::Operator *op) const
{
  // Circle Conv2D may not have a bias but we won't support this
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  if (inputs.size() != 3)
    return false;

  return true;
}

void CircleConv2DGraphBuilder::build(const circle::Operator *op, GraphBuilderContext *context) const
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

  // Conv2D node itself
  auto conv2d_node = graph->nodes()->create<CircleConv2D>();
  assert(outputs.size() > 0);
  uint32_t output_ti = static_cast<uint32_t>(outputs[0]);
  auto output_tensor = tensors->Get(output_ti);
  auto tname = tensor_name(output_tensor);
  conv2d_node->name(tname);
  auto quantization = tensor_quantization(output_tensor);
  if (quantization)
  {
    auto quantparam = luci_quantparam(quantization);
    if (quantparam.get())
      conv2d_node->quantparam(std::move(quantparam));
  }
  opfinder->enroll(conv2d_node, op);
  tensorfinder->enroll(conv2d_node, output_tensor);
  for (auto output : outputs)
  {
    INFO(l) << "[luci] NodeFinder conv2d_node(" << output << ") -> " << conv2d_node << std::endl;
    nodefinder->enroll(output, conv2d_node);
  }
  // TODO Output Shape ?

  const auto *options = op->builtin_options_as_Conv2DOptions();

  // Padding
  auto padding = luci_padding(options->padding());
  conv2d_node->padding(padding);

  // Stride
  conv2d_node->stride()->w(options->stride_w());
  conv2d_node->stride()->h(options->stride_h());

  // Activation
  auto actfunctype = luci_actfunc(options->fused_activation_function());
  conv2d_node->fusedActivationFunction(actfunctype);

  // TODO extract function that returns CircleConst from tensor_index
  // Conv2D kernel tensor + buffer to CircleConst node
  uint32_t kernel_ti = static_cast<uint32_t>(inputs[1]);
  auto kernel_const = create_circleconst(context, kernel_ti);
  conv2d_node->filter(kernel_const);

  // Conv2D bias tensor + buffer to CircleConst node, if exist
  if (inputs.size() == 3)
  {
    uint32_t bias_ti = static_cast<uint32_t>(inputs[2]);
    auto bias_const = create_circleconst(context, bias_ti);
    conv2d_node->bias(bias_const);
  }
  else
  {
    // TODO if we should support without bias, let's implement here
  }

  // Create GraphUpdate for graph connection for Conv2D node
  auto update = stdex::make_unique<CircleConv2DGraphUpdate>(conv2d_node);
  updates->enroll(std::move(update));
}

} // namespace luci

namespace
{

void CircleConv2DGraphUpdate::update(GraphBuilderContext *context)
{
  auto opfinder = context->opfinder();
  auto nodefinder = context->nodefinder();

  auto op = opfinder->op(_node);

  // set input 'input'
  const std::vector<int32_t> &inputs = luci::as_index_vector(op->inputs());
  uint32_t idx_input = static_cast<uint32_t>(inputs[0]);
  auto node_input = nodefinder->node(idx_input);
  assert(node_input != nullptr);
  _node->input(node_input);
}

} // namespace
