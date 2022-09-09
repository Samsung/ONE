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
enum class BuilderId
{
#define REGISTER_KERNEL(builtin_operator, name) Circle##name,
#include "KernelsToBuild.lst"
  Size // casts to count of values in BuilderId enum
};
#undef REGISTER_KERNEL

/**
 * @brief Registry of kernel builders
 *
 * This class contains mapping from Opcodes to kernel builder functions
 */

class KernelBuilderRegistry
{
public:
  using KernelBuilderFunc = std::unique_ptr<Kernel>(
    std::vector<std::pair<const Tensor *, int32_t>> &, std::vector<std::pair<Tensor *, int32_t>> &,
    const uint32_t, KernelBuilder &);

  KernelBuilderRegistry()
  {
#define REGISTER_KERNEL(builtin_operator, name)                                        \
  register_kernel_builder(circle::BuiltinOperator::BuiltinOperator_##builtin_operator, \
                          build_kernel_Circle##name);

#include "KernelsToBuild.lst"

#undef REGISTER_KERNEL
  }

  KernelBuilderFunc *get_kernel_builder_func(circle::BuiltinOperator opcode) const
  {
    auto tmp = size_t(opcode);
    return _operator_builders.at(size_t(opcode));
  }

private:
  std::map<int32_t, KernelBuilderFunc *> _operator_builders;

  void register_kernel_builder(circle::BuiltinOperator id, KernelBuilderFunc *func)
  {
    _operator_builders[size_t(id)] = func;
  }
};

KernelBuilder::KernelBuilder(RuntimeGraph *runtime_graph, CircleReader *circle_reader)
  : _runtime_graph(runtime_graph), _circle_reader(circle_reader)
{
  _builder_registry = std::make_unique<KernelBuilderRegistry>();
}

KernelBuilder::~KernelBuilder()
{
  // Need to define in this CPP to hide KernelBuilderRegistry internals.
  // This destructor deletes _builder_registry
}

std::unique_ptr<Kernel>
KernelBuilder::build(std::vector<std::pair<const Tensor *, int32_t>> &inputs,
                     std::vector<std::pair<Tensor *, int32_t>> &outputs, const uint32_t op_index)
{
  const auto op = _circle_reader->operators()[op_index];
  const auto opcode = _circle_reader->builtin_code(op);
  auto specific_builder = _builder_registry->get_kernel_builder_func(opcode);
  if (specific_builder != nullptr)
    return specific_builder(inputs, outputs, op_index, *this);

  std::string msg = "Unsupported operator: ";
  msg += std::to_string(static_cast<uint32_t>(opcode));
  throw std::invalid_argument(msg.c_str());
}

} // namespace luci_interpreter
