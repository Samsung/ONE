/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_KERNELS_SVDF_H
#define LUCI_INTERPRETER_KERNELS_SVDF_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class SVDF : public KernelWithParams<SVDFParams>
{
public:
  SVDF(const Tensor *input, const Tensor *weight_feature, const Tensor *weight_time,
       const Tensor *bias, const Tensor *input_activation_state, Tensor *output,
       Tensor *scratchpad_activation_state, Tensor *scratchpad_1, Tensor *scratchpad_2,
       Tensor *scratchpad_3, Tensor *scratchpad_4, Tensor *scratchpad_5, Tensor *scratchpad_6,
       const SVDFParams &params);

  const Tensor *input() const { return _inputs[0]; }
  const Tensor *weight_feature() const { return _inputs[1]; }
  const Tensor *weight_time() const { return _inputs[2]; }
  const Tensor *bias() const { return _inputs[3]; }
  const Tensor *input_activation_state() const { return _inputs[4]; }

  Tensor *output() const { return _outputs[0]; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;
  void evalInteger() const;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_SVDF_H
