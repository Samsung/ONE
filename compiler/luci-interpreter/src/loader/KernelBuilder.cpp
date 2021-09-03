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

#include "loader/KernelBuilder.h"
#include "loader/nodes/Builders.h"

#include <stdexcept>

namespace luci_interpreter
{

#define CIRCLE_NODE(OPCODE, CLASS) CLASS,
#define CIRCLE_VNODE(OPCODE, CLASS) CLASS,

// This enum is auxiliary.
// It is duplicate of luci::CircleOpcode but initialized with CLASS instead of OPCODE,
// because list of target operators is in format of CLASS names
enum class BuilderId
{
#include <luci/IR/CircleNodes.lst>
};

#undef CIRCLE_VNODE
#undef CIRCLE_NODE

/**
 * @brief Registry of kernel builders
 *
 * This class contains mapping from Opcodes to kernel builder functions
 */

class KernelBuilderRegistry
{
public:
  using KernelBuilderFunc = std::unique_ptr<Kernel>(const luci::CircleNode *,
                                                    KernelBuilderHelper &);

  KernelBuilderRegistry()
  {
#define REGISTER_KERNEL(name) \
  register_kernel_builder(BuilderId::Circle##name, build_kernel_Circle##name);

#include "KernelsToBuild.lst"

#undef REGISTER_KERNEL
  }

  KernelBuilderFunc *get_kernel_builder_func(luci::CircleOpcode opcode)
  {
    auto opcode_int = static_cast<int>(opcode);
    if (opcode_int >= static_cast<int64_t>(_operator_builders.size()))
      return nullptr;
    return _operator_builders[opcode_int];
  }

private:
  std::vector<KernelBuilderFunc *> _operator_builders;

  void register_kernel_builder(BuilderId id, KernelBuilderFunc *func)
  {
    // Using BuilderId is a duplicate of luci::CirclreOpcode,
    // static_cast<int>(id) is equal to static_cast<int>(corresponding operation opcode).
    auto opcode = static_cast<int>(id);
    if (opcode >= static_cast<int64_t>(_operator_builders.size()))
      _operator_builders.resize(opcode + 1);
    _operator_builders[opcode] = func;
  }
};

KernelBuilder::KernelBuilder(
  const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
  const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
  : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
{
  _builder_registry = std::make_unique<KernelBuilderRegistry>();
}

KernelBuilder::~KernelBuilder()
{
  // Need to define in this CPP to hide KernelBuilderRegistry internals.
  // This destructor deletes _builder_registry
}

std::unique_ptr<Kernel> KernelBuilder::build(const luci::CircleNode *node)
{
  auto specific_builder = _builder_registry->get_kernel_builder_func(node->opcode());
  if (specific_builder != nullptr)
    return specific_builder(node, *this);

  std::string msg = "Unsupported operator: ";
  msg += std::to_string(static_cast<uint32_t>(node->opcode())) + " " + std::string(node->name());
  throw std::invalid_argument(msg.c_str());
}

} // namespace luci_interpreter
