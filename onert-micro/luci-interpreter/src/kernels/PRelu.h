/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_KERNELS_PRELU_H
#define LUCI_INTERPRETER_KERNELS_PRELU_H

#include "core/Kernel.h"
#include <vector>

namespace luci_interpreter
{
namespace kernels
{

class ChannelQuantMultipliers;

class PRelu : public Kernel
{
public:
  PRelu(const Tensor *input, const Tensor *alpha, Tensor *output);

  ~PRelu();

  const Tensor *input() const { return _inputs[0]; }
  const Tensor *alpha() const { return _inputs[1]; }
  Tensor *output() const { return _outputs[0]; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;
  void evalQuantized() const;
  void evalQuantizedS16() const;

private:
  std::vector<ChannelQuantMultipliers> _alpha_multipliers;
  // TODO merge this into one ChannelQuantMultiplier object
  int32_t _output_multiplier_identity = 0;
  int _output_shift_identity = 0;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_PRELU_H
