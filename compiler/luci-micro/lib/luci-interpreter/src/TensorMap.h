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

#ifndef LUCI_INTERPRETER_TENSORMAP_H
#define LUCI_INTERPRETER_TENSORMAP_H

#include "core/Tensor.h"

#include <loco/IR/Node.forward.h>

#include <cassert>
#include <unordered_map>

namespace luci_interpreter
{

  class TensorMap
  {
  public:
    // Associates a node with a tensor.
    void setTensor(const loco::Node *node, std::unique_ptr<Tensor> tensor)
    {
      assert(_tensors.find(node) == _tensors.cend());
      _tensors.emplace(node, std::move(tensor));
    }

    // Finds a tensor associated with a node.
    Tensor *getTensor(const loco::Node *node)
    {
      const auto it = _tensors.find(node);
      if (it == _tensors.cend())
      {
        return nullptr;
      }
      return it->second.get();
    }

  private:
    std::unordered_map<const loco::Node *, std::unique_ptr<Tensor>> _tensors;
  };

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_TENSORMAP_H
