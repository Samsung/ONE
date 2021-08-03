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

#ifndef LUCI_INTERPRETER_LOADER_KERNELBUILDER_H
#define LUCI_INTERPRETER_LOADER_KERNELBUILDER_H

#include "loader/KernelBuilderHelper.h"

#include "core/Kernel.h"
#include "core/RuntimeGraph.h"

#include <luci/IR/CircleNodeVisitor.h>

#include <memory>
#include <unordered_map>

namespace luci_interpreter
{

#define REGISTER_KERNEL(name)                                                            \
  std::unique_ptr<Kernel> build_kernel_Circle##name(const luci::CircleNode *circle_node, \
                                                    KernelBuilderHelper &helper);

#include "KernelsToBuild.lst"

#undef REGISTER_KERNEL

class KernelBuilderRegistry;

class KernelBuilder : public KernelBuilderHelper
{
public:
  KernelBuilder(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor);

  ~KernelBuilder();

  std::unique_ptr<Kernel> build(const luci::CircleNode *node);

private:
  std::unique_ptr<KernelBuilderRegistry> _builder_registry;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_KERNELBUILDER_H
