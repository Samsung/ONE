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

#include <luci/IR/CircleOpcode.h>
#include <luci/IR/CircleNodeDecl.h>

#include <stdexcept>

namespace
{

// TODO Extract this helper function
const std::string toString(luci::CircleOpcode opcode)
{
  static const char *names[] = {
#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) #CIRCLE_CLASS,
#define CIRCLE_VNODE(OPCODE, CIRCLE_CLASS) #CIRCLE_CLASS,
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE
#undef CIRCLE_VNODE
  };

  auto const node_name = names[static_cast<int>(opcode)];

  assert(std::string(node_name).substr(0, 6) == "Circle"); // FIX_ME_UNLESS

  // Return substring of class name ("Circle" is sliced out)
  // Ex: Return "Conv2D" for "CircleConv2D" node
  return std::string(node_name).substr(6);
}

} // namespace

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
  Size // casts to count of values in BuilderId enum
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

  KernelBuilderRegistry() : _operator_builders(size_t(BuilderId::Size), nullptr)
  {
#define REGISTER_KERNEL(name) \
  register_kernel_builder(BuilderId::Circle##name, build_kernel_Circle##name);

#include "KernelsToBuild.lst"

#undef REGISTER_KERNEL
  }

  KernelBuilderFunc *get_kernel_builder_func(luci::CircleOpcode opcode) const
  {
    return _operator_builders.at(size_t(opcode));
  }

private:
  std::vector<KernelBuilderFunc *> _operator_builders;

  void register_kernel_builder(BuilderId id, KernelBuilderFunc *func)
  {
    // Using BuilderId is a duplicate of luci::CirclreOpcode,
    // size_t(id) is equal to size_t(corresponding operation opcode).
    assert(size_t(id) < _operator_builders.size());
    _operator_builders[size_t(id)] = func;
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
  msg += toString(node->opcode()) + " in " + std::string(node->name());
  throw std::invalid_argument(msg.c_str());
}

} // namespace luci_interpreter
