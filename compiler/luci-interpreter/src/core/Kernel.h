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

#ifndef LUCI_INTERPRETER_CORE_KERNEL_H
#define LUCI_INTERPRETER_CORE_KERNEL_H

#include "luci_interpreter/core/Tensor.h"

#include <vector>

namespace luci_interpreter
{

// Base class for all kernels.
class Kernel
{
protected:
  Kernel(std::vector<const Tensor *> inputs, std::vector<Tensor *> outputs)
    : _inputs(std::move(inputs)), _outputs(std::move(outputs))
  {
  }

public:
  virtual ~Kernel() = default;

  std::vector<const Tensor *> getInputTensors() const { return _inputs; }
  std::vector<Tensor *> getOutputTensors() const { return _outputs; }

  // Configures the kernel.
  // This function is currently called once for each kernel during interpreter construction,
  // which makes it a convenient place for preparing (resizing) output tensors.
  virtual void configure() = 0;

  // Executes the kernel.
  virtual void execute() const = 0;

protected:
  // NOTE Prefer not to use these in derived classes.
  const std::vector<const Tensor *> _inputs;
  const std::vector<Tensor *> _outputs;
};

// Base class for kernels with parameters.
template <typename Params> class KernelWithParams : public Kernel
{
protected:
  KernelWithParams(std::vector<const Tensor *> inputs, std::vector<Tensor *> outputs,
                   const Params &params)
    : Kernel(std::move(inputs), std::move(outputs)), _params(params)
  {
  }

public:
  const Params &params() const { return _params; }

protected:
  const Params _params;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_KERNEL_H
