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

#include "MinMaxObserver.h"

#include <luci/IR/CircleOpcode.h>

using DataType = luci_interpreter::DataType;

namespace record_minmax
{

// postTensorWrite is only called for a node producing a tensor
void MinMaxObserver::postTensorWrite(const luci::CircleNode *node,
                                     const luci_interpreter::Tensor *tensor)
{
  // CircleOutput does not produce a tensor
  assert(node->opcode() != luci::CircleOpcode::CIRCLEOUTPUT);

  // Operators with multiple outputs
  assert(node->opcode() != luci::CircleOpcode::IF);
  assert(node->opcode() != luci::CircleOpcode::SPLIT);
  assert(node->opcode() != luci::CircleOpcode::SPLIT_V);
  assert(node->opcode() != luci::CircleOpcode::TOPK_V2);
  assert(node->opcode() != luci::CircleOpcode::UNPACK);
  assert(node->opcode() != luci::CircleOpcode::WHILE);

  if (node->opcode() == luci::CircleOpcode::CONST)
  {
    // node is not activation. Do nothing.
    return;
  }

  if (node->opcode() == luci::CircleOpcode::ARG_MAX)
  {
    // Output of arg_max is the index of the largest value across axes of a tensor
    // this should not be quantized
    return;
  }

  // Only support recording of float32 values
  if (tensor->element_type() != DataType::FLOAT32)
    throw std::runtime_error("Tensor's data type is not float");

  const auto data = tensor->data<float>();
  const auto num_elements = tensor->shape().num_elements();

  std::vector<float> buf(data, data + num_elements);
  auto minmax = std::minmax_element(buf.begin(), buf.end());
  float min = *minmax.first;
  float max = *minmax.second;

  _minmax_data.recordMinMax(node, min, max);
}

} // namespace record_minmax
