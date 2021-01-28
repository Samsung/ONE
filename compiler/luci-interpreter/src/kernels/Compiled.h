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

#ifndef LUCI_INTERPRETER_KERNELS_COMPILED_H
#define LUCI_INTERPRETER_KERNELS_COMPILED_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class Compiled : public KernelWithParams<CompiledParams>
{
public:
  Compiled(std::vector<const Tensor *> inputs, std::vector<Tensor *> output, const CompiledParams &params);

  const Tensor *input(int i) const { return _inputs[i]; }
  size_t num_inputs() const { return _inputs.size(); }
  Tensor *output(int i) const { return _outputs[i]; }
  size_t num_outputs() const { return _outputs.size(); }

  void configure() override;
  void execute() const override;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_COMPILED_H
