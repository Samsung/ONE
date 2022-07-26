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

#ifndef LUCI_INTERPRETER_LOADER_RUNTIMETOIR_H
#define LUCI_INTERPRETER_LOADER_RUNTIMETOIR_H

#include "luci_interpreter/core/Tensor.h"

#include <luci/IR/CircleNode.h>

#include <unordered_map>

namespace luci_interpreter
{

// Maps runtime entities back to IR entities. It is used to implement observing functionality.
struct RuntimeToIR
{
  std::unordered_map<const Tensor *, const luci::CircleNode *> tensor_to_node;
  std::unordered_map<const Kernel *, const luci::CircleNode *> kernel_to_node;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_RUNTIMETOIR_H
