/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "record-hessian/HessianObserver.h"

namespace record_hessian
{

void HessianObserver::postTensorWrite(const luci::CircleNode *node,
                                      const luci_interpreter::Tensor *tensor)
{
  assert(node != nullptr);
  assert(tensor != nullptr);

  auto node_outputs = loco::succs(node);
  for (auto node_output : node_outputs)
  {
    auto cur_node = dynamic_cast<luci::CircleNode *>(node_output);
    if (cur_node == nullptr)
    {
      throw std::runtime_error("Record Hessian: node output shouldn't be null.");
    }
    // TODO : ADD TCONV/DepthCONV cases
    if (cur_node->opcode() == luci::CircleOpcode::FULLY_CONNECTED ||
        cur_node->opcode() == luci::CircleOpcode::CONV_2D)
    {
      _hessian_computer.recordHessian(cur_node, tensor);
    }
  }
}

} // namespace record_hessian
