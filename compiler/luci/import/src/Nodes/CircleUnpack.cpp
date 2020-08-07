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

#include "luci/Import/Nodes/CircleUnpack.h"

#include <luci/IR/Nodes/CircleUnpack.h>
#include <luci/IR/Nodes/CircleUnpackOut.h>

#include <luci/UserSettings.h>
#include <luci/Log.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleUnpackGraphBuilder::validate(const ValidateArgs &args) const
{
  LOGGER(l);

  auto settings = luci::UserSettings::settings();

  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  const auto *options = args.op.builtin_options.AsUnpackOptions();

  if (inputs.size() != 1)
    return false;

  // NOTE real models may have mismatch
  if (static_cast<int32_t>(outputs.size()) != options->num)
  {
    if (settings->get(luci::UserSettings::Key::DisableValidation))
    {
      const auto &tensors = args.reader.tensors();
      const circle::TensorT &output_tensor = *tensors[outputs[0]];
      auto name = tensor_name(output_tensor);
      WARN(l) << "Warning: import Unpack(" << name << ") 'num' is not same as outputs used";
    }
    else
      return false;
  }

  if (options->num < 0)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &tensor = tensors.at(inputs.at(0));
  const auto &shape = tensor->shape;
  auto shape_size = static_cast<int32_t>(shape.size());
  if (shape_size > 0)
  {
    // NOTE for unknown shape, shape_size is 0
    if (options->axis < -shape_size || options->axis >= shape_size)
      return false;
  }

  return true;
}

/**
 * @brief  Unpack Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple CircleUnpackOut nodes to emulate this
 *         For two outputs that may look like this
 *
 *         --- CircleUnpack --- FullyConnected ---
 *                           \- FullyConnected ---
 *
 *         will be created like this
 *
 *         --- CircleUnpack --- CircleUnpackOut --- FullyConnected ---
 *                           \- CircleUnpackOut --- FullyConnected ---
 */

void CircleUnpackGraphBuilder::build(const circle::OperatorT &op,
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

  // NOTE Unpack has only one input so running a loop is not necessary
  //      This is provided as a reference for other Ops as a reference
  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    input_nodes.push_back(context->nodefinder()->node(input_tensor_index));
  }

  // Create CircleUnpack
  CircleUnpack *node = graph->nodes()->create<CircleUnpack>();
  node->value(input_nodes[0]);

  const auto *options = op.builtin_options.AsUnpackOptions();
  node->num(options->num);
  node->axis(options->axis);

  assert(outputs.size() > 0);
  {
    // Let's use name of output 0 as Unpack name
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));
    node->op_version(opcodes[op.opcode_index].get()->version);

    // NOTE We don't set quantization for Unpack itself but to virtual outputs
  }

  // Create virtual outputs of Unpack
  for (int32_t n = 0; n < options->num; ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleUnpackOut>();
    copy_tensor_attributes(output_tensor, nodeout);
    // mark shape_status
    if (tensors_ptr->Get(outputs[n])->shape() == nullptr)
      nodeout->shape_status(ShapeStatus::NOSHAPE);
    else
      nodeout->shape_status(ShapeStatus::VALID);

    nodeout->input(node);
    nodeout->index(n);

    context->nodefinder()->enroll(outputs[n], nodeout);
  }
}

} // namespace luci
