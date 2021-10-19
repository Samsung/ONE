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

#include "luci/Import/Nodes/CircleNonMaxSuppressionV5.h"

#include <luci/IR/Nodes/CircleNonMaxSuppressionV5.h>
#include <luci/IR/Nodes/CircleNonMaxSuppressionV5Out.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleNonMaxSuppressionV5GraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 6)
    return false;
  if (outputs.size() != 3)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &boxes_tensor = tensors.at(inputs[0]);
  if (boxes_tensor->shape.size() != 2)
    return false;
  if (boxes_tensor->shape.at(1) != 4)
    return false;
  if (boxes_tensor->shape.at(0) != tensors.at(inputs[1])->shape.at(0))
    return false;

  if (tensors.at(inputs[2])->type != circle::TensorType_INT32)
    return false;
  if (tensors.at(inputs[3])->type != circle::TensorType_FLOAT32)
    return false;
  if (tensors.at(inputs[4])->type != circle::TensorType_FLOAT32)
    return false;
  if (tensors.at(inputs[5])->type != circle::TensorType_FLOAT32)
    return false;

  return true;
}

/**
 * @brief  NonMaxSuppressionV5 Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple NonMasSuppressionV5Oout nodes to emulate this
 */

CircleNode *CircleNonMaxSuppressionV5GraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  auto node = bna.context->graph()->nodes()->create<CircleNonMaxSuppressionV5>();

  node->boxes(bna.input_nodes[0]);
  node->scores(bna.input_nodes[1]);
  node->max_output_size(bna.input_nodes[2]);
  node->iou_threshold(bna.input_nodes[3]);
  node->score_threshold(bna.input_nodes[4]);
  node->soft_nms_sigma(bna.input_nodes[5]);

  return node;
}

CircleNode *CircleNonMaxSuppressionV5GraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleNonMaxSuppressionV5Out>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
