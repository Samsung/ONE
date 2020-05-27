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

#ifndef LUCI_INTERPRETER_KERNELMAP_H
#define LUCI_INTERPRETER_KERNELMAP_H

#include "core/Kernel.h"

#include <luci/IR/CircleNode.h>

#include <cassert>
#include <unordered_map>

namespace luci_interpreter
{

class KernelMap
{
public:
  // Associates a node with the kernel.
  void setKernel(const luci::CircleNode *node, std::unique_ptr<Kernel> kernel)
  {
    assert(_kernels.find(node) == _kernels.cend());
    _kernels.emplace(node, std::move(kernel));
  }

  // Finds the kernel associated with the node.
  Kernel *getKernel(const luci::CircleNode *node)
  {
    const auto it = _kernels.find(node);
    assert(it != _kernels.cend());
    return it->second.get();
  }

private:
  std::unordered_map<const luci::CircleNode *, std::unique_ptr<Kernel>> _kernels;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELMAP_H
