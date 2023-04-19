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

#include "luci/Import/Nodes/CircleReshape.h"

#include <luci/IR/Nodes/CircleConst.h>
#include <luci/IR/Nodes/CircleReshape.h>

namespace luci
{

bool CircleReshapeGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 1 && args.op.inputs.size() != 2)
    return false;

  if (args.op.outputs.size() != 1)
    return false;

  // for two inputs, check if type is S32 or S64
  if (args.op.inputs.size() == 2)
  {
    const auto &inputs = args.op.inputs;
    const auto tensors = args.reader.tensors();
    const auto tensor_in = tensors.at(inputs.at(1));
    assert(tensor_in != nullptr);

    if (tensor_in->type() != circle::TensorType::TensorType_INT32 &&
        tensor_in->type() != circle::TensorType::TensorType_INT64)
      return false;
  }

  return true;
}

static void setup_shape_attribute(const std::vector<int32_t> &shape, CircleReshape *node)
{
  node->newShape()->rank(shape.size());
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->newShape()->dim(i) = shape[i];
  }
}

static CircleNode *create_shape_node(const std::vector<int32_t> &shape, loco::Graph *graph)
{
  auto *shape_node = graph->nodes()->create<luci::CircleConst>();
  shape_node->dtype(loco::DataType::S32);
  shape_node->rank(1);
  shape_node->dim(0) = shape.size();
  shape_node->size<loco::DataType::S32>(shape.size());
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    shape_node->at<loco::DataType::S32>(i) = shape[i];
  }
  shape_node->name("Reshape/shape");
  return shape_node;
}

CircleNode *CircleReshapeGraphBuilder::build_node(const circle::OperatorT &op,
                                                  const std::vector<CircleNode *> &inputs,
                                                  loco::Graph *graph) const
{
  // If the second input is not provided, generate it based on the value of the attribute.
  // TODO Presence of the second input is the current requirement of the IR.
  auto *shape_node = (inputs.size() == 2) ? inputs.at(1) : nullptr;
  if (shape_node == nullptr)
  {
    const auto *options = op.builtin_options.AsReshapeOptions();
    if (options != nullptr)
      shape_node = create_shape_node(options->new_shape, graph);
    else
    {
      shape_node = graph->nodes()->create<CircleOutputDummy>();
      shape_node->dtype(loco::DataType::S32);
      shape_node->rank(0);
      shape_node->name("Reshape/dummy");
    }
  }

  auto *node = graph->nodes()->create<CircleReshape>();
  node->tensor(inputs.at(0));
  node->shape(shape_node);

  const auto *options = op.builtin_options.AsReshapeOptions();
  if (options)
    setup_shape_attribute(options->new_shape, node);

  return node;
}

} // namespace luci
