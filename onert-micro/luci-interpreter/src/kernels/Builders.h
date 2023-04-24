/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_KERNELS_NODES_BUILDERS_H
#define LUCI_INTERPRETER_KERNELS_NODES_BUILDERS_H

#include "KernelBuilder.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"
#include "core/RuntimeGraph.h"

namespace luci_interpreter
{

#define REGISTER_KERNEL(builtin_operator, name)                        \
  void configure_kernel_Circle##name(const circle::Operator *cur_op,   \
                                     BaseRuntimeGraph *runtime_graph); \
                                                                       \
  void execute_kernel_Circle##name(const circle::Operator *cur_op,     \
                                   BaseRuntimeGraph *runtime_graph, bool is_inplace);

#if USE_GENERATED_LIST
#include "GeneratedKernelsToBuild.lst"
#else
#include "KernelsToBuild.lst"
#endif

#undef REGISTER_KERNEL

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_NODES_BUILDERS_H
