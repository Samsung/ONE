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

#ifndef LUCI_INTERPRETER_KERNELS_WHILE_H
#define LUCI_INTERPRETER_KERNELS_WHILE_H

#include "core/Kernel.h"
#include "core/RuntimeGraph.h"

namespace luci_interpreter
{
namespace kernels
{

class While : public Kernel
{
public:
  While(std::vector<const Tensor *> inputs, std::vector<Tensor *> outputs, RuntimeGraph *cond_graph,
        RuntimeGraph *body_graph);

  const Tensor *input(int index) const { return _inputs[index]; }
  Tensor *output(int index) const { return _outputs[index]; }

  void configure() override;
  void execute() const override;

private:
  RuntimeGraph *const _cond_graph = nullptr;
  RuntimeGraph *const _body_graph = nullptr;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_WHILE_H
