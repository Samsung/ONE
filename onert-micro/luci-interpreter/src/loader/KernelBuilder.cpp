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

namespace luci_interpreter
{
enum class BuilderId
{
#define REGISTER_KERNEL(builtin_operator, name) Circle##name,
#if USE_GENERATED_LIST
#include "GeneratedKernelsToBuild.lst"
#else
#include "KernelsToBuild.lst"
#endif
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
  using KernelBuilderFunc = std::unique_ptr<Kernel>(std::vector<const Tensor *> &&,
                                                    std::vector<Tensor *> &&, const uint32_t,
                                                    KernelBuilder &);

  KernelBuilderRegistry()
  {
#define REGISTER_KERNEL(builtin_operator, name)                                        \
  register_kernel_builder(circle::BuiltinOperator::BuiltinOperator_##builtin_operator, \
                          build_kernel_Circle##name);

#if USE_GENERATED_LIST
#include "GeneratedKernelsToBuild.lst"
#else
#include "KernelsToBuild.lst"
#endif

#undef REGISTER_KERNEL
  }

  KernelBuilderFunc *get_kernel_builder_func(circle::BuiltinOperator opcode) const
  {
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

std::unique_ptr<Kernel> KernelBuilder::build(std::vector<const Tensor *> &&inputs,
                                             std::vector<Tensor *> &&outputs,
                                             const circle::BuiltinOperator opcode,
                                             const int32_t op_index)
{
  auto specific_builder = _builder_registry->get_kernel_builder_func(opcode);
  if (specific_builder != nullptr)
    return specific_builder(std::move(inputs), std::move(outputs), op_index, *this);

  assert(false && "Unsupported operator");
}

} // namespace luci_interpreter
