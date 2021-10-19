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

#include "luci/Import/Nodes/CircleCast.h"

#include <luci/IR/Nodes/CircleCast.h>

#include <luci/UserSettings.h>
#include <luci/Log.h>

#include <loco.h>

namespace luci
{

bool CircleCastGraphBuilder::validate(const ValidateArgs &args) const
{
  LOGGER(l);

  if (!GraphBuilder::validate(args, 1))
    return false;

  auto settings = luci::UserSettings::settings();

  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  // NOTE real models do have type mismatch
  const auto *options = args.op.builtin_options.AsCastOptions();
  if (options != nullptr)
  {
    const auto &tensors = args.reader.tensors();
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    auto name = tensor_name(output_tensor);

    const auto &tensor_in = tensors.at(inputs.at(0));
    if (tensor_in->type != options->in_data_type)
    {
      if (settings->get(luci::UserSettings::Key::DisableValidation))
      {
        WARN(l) << "Warning: import Cast(" << name << ") dtype mismatch";
      }
      else
        return false;
    }
    const auto &tensor_out = tensors.at(outputs[0]);
    if (tensor_out->type != options->out_data_type)
    {
      if (settings->get(luci::UserSettings::Key::DisableValidation))
      {
        WARN(l) << "Warning: import Cast(" << name << ") dtype mismatch";
      }
      else
        return false;
    }
  }

  return true;
}

CircleNode *CircleCastGraphBuilder::build_node(const circle::OperatorT &op,
                                               const std::vector<CircleNode *> &inputs,
                                               loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleCast>();
  node->x(inputs.at(0));

  const auto *options = op.builtin_options.AsCastOptions();
  if (options != nullptr)
  {
    node->in_data_type(luci_datatype(options->in_data_type));
    node->out_data_type(luci_datatype(options->out_data_type));
  }
  else
  {
    node->in_data_type(inputs.at(0)->dtype());
    node->out_data_type(loco::DataType::Unknown);
    // type inference should use node->dtype() for Unknown
    // export should use BuiltinOptions_NONE for Unknown
  }

  return node;
}

} // namespace luci
