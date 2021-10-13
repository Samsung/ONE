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

#include "luci/Import/Nodes/CircleNonMaxSuppressionV4.h"

#include <luci/IR/Nodes/CircleNonMaxSuppressionV4.h>
#include <luci/IR/Nodes/CircleNonMaxSuppressionV4Out.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleNonMaxSuppressionV4GraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 5)
    return false;
  if (outputs.size() != 2)
    return false;

  const auto tensors = args.reader.native_tensors();
  const auto boxes_tensor = tensors.at(inputs[0]);
  const auto boxes_tensor_shape = wrap(boxes_tensor->shape());
  if (boxes_tensor_shape.size() != 2)
    return false;
  if (boxes_tensor_shape.at(1) != 4)
    return false;
  if (boxes_tensor_shape.at(0) != wrap(tensors.at(inputs[1])->shape()).at(0))
    return false;

  if (tensors.at(inputs[2])->type() != circle::TensorType_INT32)
    return false;
  if (tensors.at(inputs[3])->type() != circle::TensorType_FLOAT32)
    return false;
  if (tensors.at(inputs[4])->type() != circle::TensorType_FLOAT32)
    return false;

  return true;
}

/**
 * @brief  NonMaxSuppressionV4 Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple NonMasSuppressionV4Oout nodes to emulate this
 */

CircleNode *CircleNonMaxSuppressionV4GraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  auto node = bna.context->graph()->nodes()->create<CircleNonMaxSuppressionV4>();

  node->boxes(bna.input_nodes[0]);
  node->scores(bna.input_nodes[1]);
  node->max_output_size(bna.input_nodes[2]);
  node->iou_threshold(bna.input_nodes[3]);
  node->score_threshold(bna.input_nodes[4]);

  return node;
}

CircleNode *CircleNonMaxSuppressionV4GraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleNonMaxSuppressionV4Out>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
