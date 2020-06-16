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

#ifndef LUCI_INTERPRETER_KERNELS_AVERAGEPOOL2D_H
#define LUCI_INTERPRETER_KERNELS_AVERAGEPOOL2D_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class AveragePool2D : public KernelWithParams<Pool2DParams>
{
public:
  AveragePool2D(const Tensor *input, Tensor *output, const Pool2DParams &params);

  std::vector<const Tensor *> getInputTensors() const override { return {_input}; }
  std::vector<Tensor *> getOutputTensors() const override { return {_output}; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;
  void evalQuantized() const;

private:
  const Tensor *const _input;
  Tensor *const _output;
  int32_t _padding_height{};
  int32_t _padding_width{};
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_AVERAGEPOOL2D_H
