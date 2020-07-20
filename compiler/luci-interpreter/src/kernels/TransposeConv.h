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

#ifndef LUCI_INTERPRETER_KERNELS_TRANSPOSECONV_H
#define LUCI_INTERPRETER_KERNELS_TRANSPOSECONV_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class TransposeConv : public KernelWithParams<TransposeConvParams>
{
public:
  TransposeConv(const Tensor *output_shape, const Tensor *filter, const Tensor *input,
                Tensor *output, const TransposeConvParams &params);

  const Tensor *output_shape() const { return _inputs[0]; }
  const Tensor *filter() const { return _inputs[1]; }
  const Tensor *input() const { return _inputs[2]; }
  Tensor *output() const { return _outputs[0]; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;
  void evalQuantized() const;

private:
  std::unique_ptr<Tensor> _scratch_tensor;

  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t _output_multiplier = 0;
  int _output_shift = 0;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_TRANSPOSECONV_H
