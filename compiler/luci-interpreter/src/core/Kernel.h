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

namespace luci_interpreter
{

// Base class for all kernels.
class Kernel
{
public:
  virtual ~Kernel() = default;

  // Configures the kernel.
  // This function is currently called once for each kernel during interpreter construction,
  // which makes it a convenient place for preparing (resizing) output tensors.
  virtual void configure() = 0;

  // Executes the kernel.
  virtual void execute() const = 0;
};

// Base class for kernels with parameters.
template <typename Params> class KernelWithParams : public Kernel
{
public:
  explicit KernelWithParams(const Params &params) : _params(params) {}

  const Params &params() const { return _params; }

protected:
  const Params _params;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_KERNEL_H
