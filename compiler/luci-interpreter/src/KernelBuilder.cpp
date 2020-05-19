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

#include "KernelBuilder.h"

#include "kernels/FullyConnected.h"
#include "kernels/Softmax.h"

#include <stdexcept>

namespace luci_interpreter
{

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleConst *)
{
  throw std::runtime_error("Const node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleFullyConnected *node)
{
  assert(node->arity() == 3);

  const Tensor *input = getInputTensor(node->input());
  const Tensor *filter = getInputTensor(node->weights());
  const Tensor *bias = getOptionalInputTensor(node->bias());
  Tensor *output = getOutputTensor(node);

  FullyConnectedParams params{};
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::FullyConnected>(input, filter, bias, output, params);
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleInput *)
{
  throw std::runtime_error("Input node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleOutput *)
{
  throw std::runtime_error("Output node cannot be executed.");
}

std::unique_ptr<Kernel> KernelBuilder::visit(const luci::CircleSoftmax *node)
{
  assert(node->arity() == 1);

  const Tensor *input = getInputTensor(node->logits());
  Tensor *output = getOutputTensor(node);

  SoftmaxParams params{};
  params.beta = node->beta();

  return std::make_unique<kernels::Softmax>(input, output, params);
}

} // namespace luci_interpreter
