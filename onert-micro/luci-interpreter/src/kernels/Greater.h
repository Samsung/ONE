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

#ifndef LUCI_INTERPRETER_KERNELS_GREATER_H
#define LUCI_INTERPRETER_KERNELS_GREATER_H

#include "core/Kernel.h"

namespace luci_interpreter
{
namespace kernels
{

class Greater : public Kernel
{
public:
  Greater(const Tensor *x, const Tensor *y, Tensor *output);

  const Tensor *x() const { return _inputs[0]; }
  const Tensor *y() const { return _inputs[1]; }
  Tensor *output() const { return _outputs[0]; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;
  template <typename T> void evalInteger() const;
  void evalQuantized() const;

private:
  int32_t _x_multiplier = 0;
  int _x_shift = 0;
  int32_t _y_multiplier = 0;
  int _y_shift = 0;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_GREATER_H
