/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_KERNELS_SPLIT_V_H
#define LUCI_INTERPRETER_KERNELS_SPLIT_V_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class SplitV : public Kernel
{
public:
  SplitV(const Tensor *input, const Tensor *size_splits, const Tensor *axis,
         std::vector<Tensor *> outputs);

  const Tensor *input() const { return _inputs[0]; }
  const Tensor *size_splits() const { return _inputs[1]; }
  const Tensor *axis() const { return _inputs[2]; }
  Tensor *output(int index) const { return _outputs[index]; }

  void configure() override;
  void execute() const override;

private:
  int32_t _axis_value{};
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_SPLIT_V_H
