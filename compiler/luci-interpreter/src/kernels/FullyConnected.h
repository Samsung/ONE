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

#ifndef LUCI_INTERPRETER_KERNELS_FULLYCONNECTED_H
#define LUCI_INTERPRETER_KERNELS_FULLYCONNECTED_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class FullyConnected : public KernelWithParams<FullyConnectedParams>
{
public:
  FullyConnected(const Tensor *input, const Tensor *weights, const Tensor *bias, Tensor *output,
                 const FullyConnectedParams &params);

  std::vector<const Tensor *> getInputTensors() const override { return {_input, _weights, _bias}; }
  std::vector<Tensor *> getOutputTensors() const override { return {_output}; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;

private:
  const Tensor *const _input;
  const Tensor *const _weights;
  const Tensor *const _bias;
  Tensor *const _output;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_FULLYCONNECTED_H
